import torch
from torchvision.models.feature_extraction import get_graph_node_names

from simple_kd.utils import FeatureSelector, apply_hooks

if __name__ == "__main__":
    from torchvision.models import resnet18  # , resnet34, mobilenet_v3_small

    model = resnet18()
    _, eval_nodes = get_graph_node_names(model)
    selector = FeatureSelector(eval_nodes)
    vals = selector.extract()
    print(f"The selected layers are: {vals}")
    features = apply_hooks(model=model, layers=vals or [])

    dummy_images1 = torch.randn(4, 3, 224, 224)
    dummy_images2 = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_images1)
        old_feat = features[0].clone()
        output2 = model(dummy_images2)
        print(
            f"If this is false, then hooks are working and getting updated: {torch.equal(old_feat, features[0])}"
        )
    print(features[0].shape, features[1].shape, features[2].shape)
