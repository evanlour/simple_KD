import torch
from accelerate import Accelerator
import inspect
from simple_kd.utils import SimpleGradClip

def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """Passes only kwargs that fn's signature actually accepts, ignoring None."""
    target = getattr(fn, "forward", fn)
    sig = inspect.signature(target)
    valid_names = set(sig.parameters.keys())
    return {
        k: v for k, v in kwargs.items()
        if k in valid_names and v is not None
    }

def images_from_generator(generator: torch.nn.Module,
                          augmentations: torch.nn.Module,
                          device: torch.device,
                          batch_size: int,
                          generator_latent_size: int,
                          requires_grad: bool=False):
    z = torch.randn(batch_size, generator_latent_size, device=device)
    if requires_grad:
        generator.train()
        gen_imgs = generator(z)
        gen_imgs = (gen_imgs + 1.0) / 2.0
        gen_imgs = augmentations(gen_imgs)
    else:
        generator.eval()
        with torch.no_grad():
            gen_imgs = generator(z)
            gen_imgs = (gen_imgs + 1.0) / 2.0
            gen_imgs = augmentations(gen_imgs)
    return gen_imgs

def train_model_with_generator(accelerator: Accelerator,
                                images: torch.Tensor,
                                student_model: torch.nn.Module,
                                teacher_model: torch.nn.Module,
                                generator: torch.nn.Module,
                                criterion: torch.nn.Module,
                                student_optimizer: torch.optim.Optimizer,
                                student_stages: list[torch.Tensor] | tuple[torch.Tensor] =[],
                                teacher_stages: list[torch.Tensor] | tuple[torch.Tensor] =[],
                                clipper: SimpleGradClip | None = None,):

    train_loss = torch.tensor(0.0, device=accelerator.device)
    generator.eval()
    student_model.train()
    with accelerator.accumulate(student_model):
        student_optimizer.zero_grad(set_to_none=True)
        student_output = student_model(images)
        with torch.no_grad():
            teacher_output = teacher_model(images)
        available_args = {
            "student_output": student_output,
            "teacher_output": teacher_output.detach() if teacher_output is not None else None,
            "student_stages": student_stages,
            "teacher_stages": [v.detach() for v in teacher_stages] if teacher_stages is not None else None,
        }
        filtered_args = _filter_kwargs_for_callable(criterion, available_args)
        train_loss = criterion(**filtered_args)
        accelerator.backward(train_loss)

        if clipper is not None:
            accelerator.unscale_gradients(student_optimizer)
            clipper(student_model.parameters())
        student_optimizer.step()
    return train_loss


def train_generator(accelerator: Accelerator,
                                images: torch.Tensor,
                                student_model: torch.nn.Module,
                                teacher_model: torch.nn.Module,
                                generator: torch.nn.Module,
                                criterion: torch.nn.Module,
                                generator_optimizer: torch.optim.Optimizer,
                                student_stages: list[torch.Tensor] | tuple[torch.Tensor] =[],
                                teacher_stages: list[torch.Tensor] | tuple[torch.Tensor] =[],
                                clipper: SimpleGradClip | None = None,):

    train_loss = torch.tensor(0.0, device=accelerator.device)
    generator.train()
    student_model.eval()
    with accelerator.accumulate(generator):
        generator_optimizer.zero_grad(set_to_none=True)
        student_output = student_model(images)
        with torch.no_grad():
            teacher_output = teacher_model(images)
        available_args = {
            "student_output": student_output,
            "teacher_output": teacher_output.detach() if teacher_output is not None else None,
            "student_stages": student_stages,
            "teacher_stages": [v.detach() for v in teacher_stages] if teacher_stages is not None else None,
        }
        filtered_args = _filter_kwargs_for_callable(criterion, available_args)
        train_loss = criterion(**filtered_args)
        accelerator.backward(train_loss)

        if clipper is not None:
            accelerator.unscale_gradients(generator_optimizer)
            clipper(student_model.parameters())
        generator_optimizer.step()
    return train_loss
