from abc import ABC, abstractmethod


class GenerationInterface(ABC):
    @staticmethod
    @abstractmethod
    def generate_image_with_text2img(*, request) -> str:
        raise NotImplementedError("generate_image_with_text2img method is not implemented")