from .trainer import RegularTrainer, DpsgdTrainer, DpsgdFTrainer, DpsgdGlobalTrainer, DpsgdGlobalAdaptiveTrainer


def create_trainer(
        train_loader,
        valid_loader,
        test_loader,
        model,
        optimizer,
        privacy_engine,
        evaluator,
        writer,
        device,
        config
):
    kwargs = {
        'method': config['method'],
        'max_epochs': config['max_epochs'],
        'num_groups': config['num_groups'],
        'selected_groups': config['selected_groups'],
        'evaluate_angles': config['evaluate_angles'],
        'evaluate_hessian': config['evaluate_hessian'],
        'angle_comp_step': config['angle_comp_step'],
        'lr': config['lr'],
        'seed': config['seed'],
        'num_hutchinson_estimates': config['num_hutchinson_estimates'],
        'sampled_expected_loss': config['sampled_expected_loss']
    }

    if config["method"] == "regular":
        trainer = RegularTrainer(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )
    elif config["method"] == "dpsgd":
        trainer = DpsgdTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            **kwargs
        )
    elif config["method"] == "dpsgd-f":
        trainer = DpsgdFTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            base_max_grad_norm=config["base_max_grad_norm"],  # C0
            counts_noise_multiplier=config["counts_noise_multiplier"],  # noise scale applied on mk and ok
            **kwargs
        )
    elif config["method"] == "dpsgd-global":
        trainer = DpsgdGlobalTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            strict_max_grad_norm=config["strict_max_grad_norm"],
            **kwargs
        )
    elif config["method"] == "dpsgd-global-adapt":
        trainer = DpsgdGlobalAdaptiveTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            strict_max_grad_norm=config["strict_max_grad_norm"],
            bits_noise_multiplier=config["bits_noise_multiplier"],
            lr_Z=config["lr_Z"],
            threshold=config["threshold"],
            **kwargs
        )
    else:
        raise ValueError("Training method not implemented")

    return trainer
