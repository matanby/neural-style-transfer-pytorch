from fire import Fire

import image_utils
from style_transfer import StyleTransfer, StyleTransferConfig


def run(content_path: str, style_path: str, output_path: str, **kwargs) -> None:
    style_transfer = StyleTransfer()
    config = StyleTransferConfig().update(**kwargs)
    content = image_utils.load(content_path)
    style = image_utils.load(style_path)
    result = style_transfer.transfer(content, style, config)
    image_utils.save(result, output_path)


if __name__ == '__main__':
    Fire(run)
