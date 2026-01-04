from src.eat import EAT



def give_model(config):
    if config.finetune.model_choose != 'EAT':
        raise ValueError(
            f"This repo release only keeps the core method: EAT. Got: {config.finetune.model_choose!r}"
        )

    if config.trainer.dataset_choose != 'EDD_seg':
        return EAT(**config.models.eat.branch1)
    return EAT(**config.models.eat.branch5)