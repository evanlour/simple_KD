import curses

import torch


class FeatureSelector:
    def __init__(self, options: list[str], title: str = "Select the intermediate layers of your model"):
        self.title = title
        self.options = options

    def _loop(self, stdscr):
        curses.curs_set(0) # Disables cursor
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()

        current_index = 0
        top_item = 0 # The first visible item index
        selected = set() # This will store the saved options

        while True:
            #stdscr.clear()
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            max_rows = max(1, h - 4)

            # Define the top item that will be displayed
            if current_index < top_item:
                top_item = current_index
            elif current_index >= top_item + max_rows:
                top_item = current_index - max_rows + 1

            stdscr.addstr(0, 2, self.title, curses.A_BOLD | curses.A_UNDERLINE)
            stdscr.addstr(1, 2, "Navigate with the arrow keys | Space: Select | Enter: Done | 'q': Abort", curses.A_DIM)

            visible_options = self.options[top_item : top_item + max_rows]
            for i, opt in enumerate(visible_options):
                is_selected = "[x]" if top_item + i in selected else "[ ]"
                display_str = f"{is_selected} {opt}"[: w - 4]

                if top_item + i == current_index:
                    stdscr.addstr(i + 3, 2, display_str, curses.A_REVERSE)
                else:
                    stdscr.addstr(i + 3, 2, display_str)

            stdscr.refresh()

            # Input management
            key = stdscr.getch()
            if key == curses.KEY_UP and current_index > 0:
                current_index -= 1
            elif key == curses.KEY_DOWN and current_index < len(self.options) - 1:
                current_index += 1
            elif key == ord(" "):
                if current_index in selected:
                    selected.remove(current_index)
                else:
                    selected.add(current_index)
            elif key in (curses.KEY_ENTER, 10, 13):
                return [self.options[i] for i in sorted(selected)]
            elif key in (ord("q"), ord("Q")):
                return

    def extract(self):
        return curses.wrapper(self._loop)

def apply_hooks(model: torch.nn.Module, layers: list[str] | tuple[str, ...], detach: bool=False) -> list[torch.Tensor]:
    features = [torch.empty(0) for _ in layers]

    for i, layer in enumerate(layers):
        current_layer = model.get_submodule(layer)

        def create_hook(index: int):
            def hook(model: torch.nn.Module, input: tuple, output: torch.Tensor):
                features[index] = output.detach() if detach else output
            return hook

        current_layer.register_forward_hook(create_hook(i))
    return features
