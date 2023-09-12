from hubs.neural_hub import Neural


N = Neural()
"""
N.run_model(
    model="perceptron_multi",
    file_name="HeartDB.xlsx",
    iter=10,
    alpha=0.3,
    test_split=0,
    norm=False,
    stop_condition=0,
    neurons=1,
    avoid_column=0
)"""

N.run_model(
    model="ffm_tf",
    file_name="HeartDB.xlsx",
    iter=500,
    alpha=0.005,
    test_split=0.1,
    norm=True,
    stop_condition=50,
    neurons=1,
    avoid_column=0,
)
