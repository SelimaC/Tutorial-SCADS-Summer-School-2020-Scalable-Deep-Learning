import argparse
import numpy as np
from set_mlp import SET_MLP
from nn_functions import *
import time


def load_fashion_mnist_data(no_training_samples, no_testing_samples):
    np.random.seed(0)

    data = np.load("data/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:no_training_samples], :]
    y_train = data["Y_train"][index_train[0:no_training_samples], :]
    x_test = data["X_test"][index_test[0:no_testing_samples], :]
    y_test = data["Y_test"][index_test[0:no_testing_samples], :]

    # normalize in 0..1
    x_train = x_train.astype('float64') / 255.
    x_test = x_test.astype('float64') / 255.

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_training_samples", default="5000", type=int,
                        help='max 60000 for Fshion MNIST')
    parser.add_argument("--no_testing_samples", default="1000", type=int,
                        help='max 10000 for Fshion MNIST')
    parser.add_argument("--no_hidden_neurons_layer", default="1000", type=int)
    parser.add_argument("--epsilon", default="13", type=int,
                        help='set the sparsity level')

    parser.add_argument("--zeta", default="0.3", type=float,
                        help='''in [0..1]. It gives the percentage of
                        unimportant connections which are removed and replaced
                        with random ones after every epoch''')
    parser.add_argument("--dropout_rate", default="0.2", type=float)
    parser.add_argument("--learning_rate", default="0.05", type=float)
    parser.add_argument("--momentum", default="0.9", type=float)
    parser.add_argument("--weight_decay", default="0.0002", type=float)

    parser.add_argument("--no_training_epochs", default="400", type=int)
    parser.add_argument("--train_batch_size", default="40", type=int)
    parser.add_argument("--test_batch_size", default="100", type=int)
    parser.add_argument("--runs", default="1", type=int)
    parser.add_argument("--save_file_path", default="Pretrained_results", 
                        type=str)

    parser.add_argument("--testing", default="True", type=bool,
                        help='''test model performance on the test data at
                            each epoch''')
    parser.add_argument("--monitor", default="True", type=bool)

    parser.add_argument("--save_model", default="True", type=bool,
                        help='whether to save the model after training')
    parser.add_argument("--save_model_loc", default="./models", type=str,
                        help='where to save the model')
    parser.add_argument("--model_name", default="SET_F_MNIST", type=str,
                        help='name to be used for saving')
    parser.add_argument("--load_model_path", default='', type=str,
                        help='Loads model from location if not empty')

    parser.add_argument("--skip_training", default=False, type=bool)

    args = parser.parse_args()

    sum_training_time = 0

    for i in range(args.runs):

        x_train, y_train, x_test, y_test = load_fashion_mnist_data(
            args.no_training_samples,
            args.no_testing_samples)

        set_mlp = SET_MLP(
            (x_train.shape[1],
             args.no_hidden_neurons_layer,
             args.no_hidden_neurons_layer,
             args.no_hidden_neurons_layer,
             y_train.shape[1]), (
            Relu, Relu, Relu, Softmax), 
            epsilon=args.epsilon, 
            load_model_path=args.load_model_path)

        start_time = time.time()
        if not args.skip_training:
            set_mlp.fit(
                x_train,
                y_train,
                x_test,
                y_test,
                loss=CrossEntropy,
                epochs=args.no_training_epochs,
                batch_size=args.train_batch_size,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                zeta=args.zeta,
                dropoutrate=args.dropout_rate,
                testing=args.testing,
                save_filename=f'''{args.save_file_path}/set_mlp_
                    {args.no_training_samples}_training_samples_e
                    {args.epsilon}_rand{i}''', monitor=args.monitor)

            step_time = time.time() - start_time
            print("\nTotal training time: ", step_time)
            sum_training_time += step_time

            # save model
            if args.save_model:
                save_loc = f'''{args.save_model_loc}/{args.model_name}_{time.time()}'''
                set_mlp.save_model(save_loc)

        # test SET-MLP
        accuracy, _ = set_mlp.predict(x_test, y_test,
                                      batch_size=args.test_batch_size)

        print('\nAccuracy of the last epoch on the testing data: ', accuracy)
    print(f'''Average training time over {args.runs}
          runs is {sum_training_time/args.runs} seconds''')