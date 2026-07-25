import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib
import mlflow
import mlflow.sklearn
import argparse
import yaml
from src.utils.logger import get_pipeline_logger

class TrainDecisionTree:
    """Model training pipeline using Decision Tree and MLflow tracking."""
    
    def __init__(
                self,
                train_data_path: str,
                test_data_path: str,
                output_path: str,
                target_column: str,
                logger: logging.Logger,
                experiment_name: str = "Decision_Tree_Training"
                ) -> None:
        """Initializes paths, logger, and MLflow tracking configurations."""
        self.train_data_path = Path(train_data_path)
        self.test_data_path = Path(test_data_path)
        self.output_path = Path(output_path)
        self.target_column = target_column
        self.logging = logger
        self.experiment_name = experiment_name
        
        mlflow.set_experiment(self.experiment_name)

    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reads selected feature train and test CSV data into DataFrames."""
        df_train = pd.read_csv(self.train_data_path)
        df_test = pd.read_csv(self.test_data_path)
        return df_train, df_test

    def train_model(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[DecisionTreeRegressor, dict]:
        """Trains the model and evaluates metrics inside an MLflow run context."""
        target_col = self.target_column
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        params = {
            "max_depth": 5,
            "min_samples_split": 2,
            "random_state": 42
        }

        with mlflow.start_run() as run:
            self.logging.info(f"Started MLflow run: {run.info.run_id}")
            
            mlflow.log_params(params)

            self.logging.info('training decision tree model...')
            model = DecisionTreeRegressor(**params)
            model.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            metrics = {"train_r2": train_score, "test_r2": test_score}
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, artifact_path="decision_tree_model")
            
            self.logging.info(f"Model trained. Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")
            
        return model, metrics

    def save_model_local(self, model: DecisionTreeRegressor, version: str) -> None:
        """Saves a local backup of the trained model file using joblib."""
        new_filename = f"decision_tree_{version}.joblib"
        destination = self.output_path / new_filename
        joblib.dump(model, destination)
        self.logging.info(f"Local model backup saved to {destination}")

    def run_step(self) -> bool:
        """Executes the training pipeline sequentially."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        df_train, df_test = self.read_data()
        model, metrics = self.train_model(df_train, df_test)
        
        self.save_model_local(model, 'trained')
        
        self.logging.info('Training process complete!')
        return True

def main(config_path: str):
    """
    Calls the SelectFeatures class

    Args:
        config_path (str): path to the configuration file for the run
    Returns:
        (bool): confirmation of success
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_output_top_level = config['run_output']['top_level']
    run_output_name = config['name']
    run_output = run_output_top_level + run_output_name

    log_dir = Path(run_output + "/logs")
    log_file = log_dir / "production.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_pipeline_logger(log_file_path=log_file)
    logger.info("Running train_dt...")

    train_data_path = run_output + config['run_output']['selected_dir_train_data']
    test_data_path = run_output + config['run_output']['selected_dir_test_data']
    output_path = run_output + config['run_output']['model_dir']
    target_column = config['feature_selection']['target_column']

    train_dt = TrainDecisionTree(
        train_data_path = train_data_path,
        test_data_path = test_data_path,
        output_path = output_path,
        target_column = target_column[0],
        logger = logger
    )
    train_dt.run_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for calling script in cli')
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise ValueError(f"Error: Configuration file not found at {config_path}")

    main(config_path)
