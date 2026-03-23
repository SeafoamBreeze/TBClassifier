import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
import mlflow
from pytorch_lightning.loggers import MLFlowLogger

from data_pipeline.data_pipeline import DataPipeline
from neural_network.densenet_classifier import DenseNetClassifier

from config import DATASET_PATH, TRACKING_DB, STUDY_DB

def objective(trial):

    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32])
    }

    with mlflow.start_run(nested=True) as run:

        mlflow.log_params({'trial_number': trial.number})
        mlflow.log_params(hyperparams)

        print(f"\n{'='*70}")
        print(f"Trial {trial.number}: {hyperparams}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"{'='*70}")

        fold_scores = {
            'val_acc': [],
            'val_loss': [],
            'val_TB_precision': [],
            'val_TB_recall': [],
            'val_mcc': [],
            'test_acc': [],
        }

        for fold_idx in range(5):
            print(f"\n--- Fold {fold_idx + 1}/5 ---")

            datamodule = DataPipeline(
                dataset_dir=DATASET_PATH,
                batch_size=hyperparams['batch_size'],
                fold_idx=fold_idx,
                n_splits=5,
                tuning = True
            )

            model = DenseNetClassifier(
                learning_rate=hyperparams['learning_rate'],
                dropout=hyperparams['dropout'],
                weight_decay=hyperparams['weight_decay']
            )

            early_stop = EarlyStopping(
                monitor='val_mcc',
                mode='max',
                patience=5,
                verbose=False
            )

            pruning = PyTorchLightningPruningCallback(trial, monitor='val_mcc')

            trainer = pl.Trainer(
                max_epochs=30,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                callbacks=[early_stop, pruning],
                enable_progress_bar=True,
                logger=MLFlowLogger(
                    experiment_name="TBClassifier_tuning",
                    tracking_uri=f"sqlite:///{TRACKING_DB}"
                ),
                deterministic=True
            )

            try:
                trainer.fit(model, datamodule=datamodule)
                val_metrics = trainer.validate(model, datamodule=datamodule, verbose=False)[0]

                fold_scores['val_acc'].append(val_metrics['val_acc'])
                fold_scores['val_loss'].append(val_metrics['val_loss'])
                fold_scores['val_TB_precision'].append(val_metrics['val_TB_precision'])
                fold_scores['val_TB_recall'].append(val_metrics['val_TB_recall'])
                fold_scores['val_mcc'].append(val_metrics['val_mcc'])

                mlflow.log_metrics({
                    f'fold_{fold_idx}_best_epoch': trainer.current_epoch,
                    f'fold_{fold_idx}_val_acc': val_metrics['val_acc'],
                    f'fold_{fold_idx}_val_loss': val_metrics['val_loss'],
                    f'fold_{fold_idx}_val_TB_precision': val_metrics['val_TB_precision'],
                    f'fold_{fold_idx}_val_TB_recall': val_metrics['val_TB_recall'],
                    f'fold_{fold_idx}_val_mcc': val_metrics['val_mcc']
                }, step=fold_idx)

                intermediate_value = np.mean(fold_scores['val_mcc'])
                trial.report(intermediate_value, step=fold_idx)

                if trial.should_prune():
                    print(f"Trial pruned at fold {fold_idx + 1}")
                    raise optuna.TrialPruned()

            except Exception as e:
                print(f"Fold {fold_idx + 1} failed: {e}")
                fold_scores['val_acc'].append(0.0)
                fold_scores['val_loss'].append(float('inf'))
                fold_scores['val_TB_precision'].append(0.0)
                fold_scores['val_TB_recall'].append(0.0)
                fold_scores['val_mcc'].append(0.0)

        mean_val_acc = np.mean(fold_scores['val_acc'])
        mean_val_loss = np.mean(fold_scores['val_loss'])
        mean_val_TB_precision = np.mean(fold_scores['val_TB_precision'])
        mean_val_TB_recall = np.mean(fold_scores['val_TB_recall'])
        mean_val_mcc = np.mean(fold_scores['val_mcc'])

        print(f"mean_val_acc: {mean_val_acc}")
        print(f"mean_val_loss: {mean_val_loss}")
        print(f"mean_val_TB_precision: {mean_val_TB_precision}")
        print(f"mean_val_TB_recall: {mean_val_TB_recall}")
        print(f"mean_val_mcc: {mean_val_mcc}")

        mlflow.log_metrics({
            'mean_val_acc': mean_val_acc,
            'mean_val_loss': mean_val_loss,
            'mean_val_TB_precision': mean_val_TB_precision,
            'mean_val_TB_recall': mean_val_TB_recall,
            'mean_val_mcc': mean_val_mcc
        })

        model_summary = str(model)
        mlflow.log_text(model_summary, f"model_summary_trial_{trial.number}.txt")

        return mean_val_mcc
    
def run_optimization(n_trials, timeout, n_splits, max_epochs_per_fold):

    print(f"\n{'='*70}")
    print("OPTUNA OPTIMIZATION SETUP")
    print(f"{'='*70}")
    print(f"run_optimization(): Trials: {n_trials}")
    print(f"run_optimization(): Timeout: {timeout}s ({timeout/3600:.1f} hours)")
    print(f"run_optimization(): CV Folds: {n_splits}")
    print(f"run_optimization(): Max epochs per fold: {max_epochs_per_fold}")
    print(f"run_optimization(): Sampler: TPE (n_startup={5}, n_candidates={24})")
    print(f"run_optimization(): Pruner: Hyperband (min_resource={3}, max_resource={max_epochs_per_fold}, reduction_factor={3})")

    study = optuna.create_study(
        study_name="TBClassifier",
        direction='maximize',
        storage=f"sqlite:///{STUDY_DB}",
        sampler=TPESampler(
            n_startup_trials=5,      # Random exploration first
            n_ei_candidates=24       # Bayesian optimization candidates
        ),
        pruner=HyperbandPruner(
            min_resource=3,            # At least 3 epochs before pruning
            max_resource=max_epochs_per_fold,  # Max epochs per trial
            reduction_factor=3         # Aggressive pruning factor
        ),
        load_if_exists=True
    )

     # Run optimization
    print(f"\n{'='*70}")
    print("STARTING OPTIMIZATION")
    print(f"{'='*70}")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,)  # Continue if single trial fails
    )

    with mlflow.start_run(run_name="STUDY_SUMMARY"):
        mlflow.log_params({
            'n_trials_requested': n_trials,
            'timeout_seconds': timeout,
            'n_splits': n_splits,
            'max_epochs_per_fold': max_epochs_per_fold,
            'sampler': 'TPE',
            'sampler_n_startup': 5,
            'sampler_n_candidates': 24,
            'pruner': 'Hyperband',
            'pruner_min_resource': 3,
            'pruner_max_resource': max_epochs_per_fold,
            'pruner_reduction_factor': 3,
        })

        mlflow.log_metrics({
            'trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'trials_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_trial_number': study.best_trial.number if study.best_trial else -1,
            'best_value': study.best_value if study.best_trial else 0.0
        })

        if study.best_trial:
            mlflow.log_dict({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_trial_number': study.best_trial.number,
            }, artifact_file="best_params.json")

    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")

    if study.best_trial:
        print(f"Best Trial: #{study.best_trial.number}")
        print(f"Best Value: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No successful trials completed!")

    return study