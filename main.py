from hubs.neural_hub import Neural


N = Neural()

N.run_model(
    model="perceptron",
    file_name="TEST_2_NEURONAS.xlsx",
    iter=1000,
    alpha=0.1,
    test_split=0,
    norm=False,
    stop_condition=0,
    neurons=2,
)
