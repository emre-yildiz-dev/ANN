import asyncio
import helpers.data_operations as do
import IO.file_io as file_io
import helpers.normalization as norm
import helpers.neural_networks as nn


async def main() -> None:
    data, headers = await file_io.read_csv_async('data/house_data.csv')
    cleaned_data = await do.remove_missing_values_async(data)
    features, target = await norm.preprocess_data_async(cleaned_data, headers.index('Price'))
    x_train, y_train, x_test, y_test = await norm.train_test_split_async(features, target)

    print("Training Features:", x_train[:5])
    print("Training Targets:", y_train[:5])
    print("Testing Targets:", x_test[:5])
    print("Testing Targets:", y_test[:5])

    # Define network parameters
    num_inputs = len(x_train[0])
    num_hidden_neurons = 10
    num_outputs = 1
    learning_rate = 0.001
    epochs = 1000

    # Initialize and train the network
    weights, biases = await nn.initialize_network(num_inputs, num_hidden_neurons, num_outputs)
    trained_weights, trained_biases = await nn.train_network(x_train, y_train, epochs, learning_rate, num_hidden_neurons)

    # Evaluate the network
    test_predictions = [await nn.forward_propagate(x, trained_weights, trained_biases) for x in x_test]
    test_loss = await nn.mean_squared_error(y_test, test_predictions)
    print("Test MSE:", test_loss)

if __name__ == '__main__':
    asyncio.run(main())
