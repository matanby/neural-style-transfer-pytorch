from fire import Fire

import image_utils
from stylizer import Stylizer, StylizerConfig


def run(content_path: str, style_path: str, output_path: str, **kwargs) -> None:
    stylizer = Stylizer()
    config = StylizerConfig().update(**kwargs)
    content = image_utils.load(content_path)
    style = image_utils.load(style_path)
    stylized = stylizer.stylize(content, style, config)
    image_utils.save(stylized, output_path)


if __name__ == '__main__':
    Fire(run)
