
import optuna

if __name__ == "__main__":


    study = optuna.study.create_study(study_name="dtstudy",
                                      directions=["minimize","minimize"],
                                      storage="postgresql://postgres@localhost",
                                      load_if_exists=True)

    for idx, trial in enumerate(study.best_trials):
        print("For Trial {}... Values: {}, Params: {}".format(trial.number, trial.values, trial.params))
