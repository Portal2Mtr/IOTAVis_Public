# Resets the postgresql to conduct a new optuna study

import optuna

optuna.delete_study(study_name="dtstudy", storage="postgresql://postgres@localhost")
