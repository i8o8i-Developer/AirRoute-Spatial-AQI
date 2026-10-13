"""
Training And Tuning Execution Driver For AirRoute Model 3.
Executes Part 1 Data Ingestion, Cross-Validation Tuning, And Evaluation Suite.
"""

from Model3.Src.TuneIdw import RunTuningAndEvaluation


def Main():
    print("==============================================================")
    print("       AirRoute Model 3 — Training And Tuning Pipeline        ")
    print("==============================================================")
    PipelineResults = RunTuningAndEvaluation()
    print("[Pipeline] Part 1 Execution Completed Successfully.")


if __name__ == "__main__":
    Main()
