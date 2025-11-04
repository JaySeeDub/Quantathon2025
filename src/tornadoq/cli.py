from __future__ import annotations
import logging
import typer
from .config import DataPaths, PreprocessConfig, CVConfig, TARGET_BINARY, TARGET_MULTI
from .io import load_excel_pair
from .preprocess import split_xy, build_preprocessor
from .benchmark_models import get_binary_estimators, get_multiclass_estimators
from .evaluate_classical_benchmarks import cv_evaluate_binary, cv_evaluate_multiclass, fit_and_eval_binary, fit_and_eval_multiclass


app = typer.Typer(no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")




@app.command()
def clean_data(train: str, test: str, scaler: str = "minmax"):
    """Quick sanity run: load and build preprocessor (no fitting)."""
    df_tr, df_te = load_excel_pair(train, test)
    pre = build_preprocessor(df_tr, PreprocessConfig(scaler=scaler))
    typer.echo(f"Train: {df_tr.shape}, Test: {df_te.shape}; Preprocessor ready.")




@app.command()
def benchmark_binary(train: str, test: str, scaler: str = "minmax", folds: int = 5, repeats: int = 1, use_smote: bool = True):
    df_tr, df_te = load_excel_pair(train, test)
    X_tr, X_te, y_tr, y_te = split_xy(df_tr, df_te, TARGET_BINARY)
    pre = build_preprocessor(X_tr, PreprocessConfig(scaler=scaler))
    models = get_binary_estimators(pre, use_smote=use_smote)
    cv_table = cv_evaluate_binary(models, X_tr, y_tr, CVConfig(folds=folds, repeats=repeats))
    best_name = cv_table.index[0]
    res = fit_and_eval_binary(models[best_name], X_tr, y_tr, X_te, y_te)
    typer.echo(cv_table)
    typer.echo({"best": best_name, **res})




@app.command()
def benchmark_multiclass(train: str, test: str, scaler: str = "minmax", folds: int = 5, repeats: int = 1, use_smote: bool = True):
    df_tr, df_te = load_excel_pair(train, test)
    X_tr, X_te, y_tr, y_te = split_xy(df_tr, df_te, TARGET_MULTI)
    pre = build_preprocessor(X_tr, PreprocessConfig(scaler=scaler))
    models = get_multiclass_estimators(pre, use_smote=use_smote)
    cv_table = cv_evaluate_multiclass(models, X_tr, y_tr, CVConfig(folds=folds, repeats=repeats))
    best_name = cv_table.index[0]
    res = fit_and_eval_multiclass(models[best_name], X_tr, y_tr, X_te, y_te)
    typer.echo(cv_table)
    typer.echo({"best": best_name, **res})




def main(): # entry point
    app()


if __name__ == "__main__":
    main()