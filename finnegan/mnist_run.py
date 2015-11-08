import csv

from network import Network


if __name__ == '__main__':
    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        t = list(reader)
        train = [[int(x) for x in y] for y in t[1:]]

    # print(train_set[1][0], train_set[1][1:])
    with open('test.csv', 'r') as f:
        reader = csv.reader(f)
        raw_nums = list(reader)
        test_set = [[int(x) for x in y] for y in raw_nums[1:]]

    ans_train = [x[0] for x in train]
    train_set = [x[1:] for x in train]
    ans_train.pop(0)
    train_set.pop(0)



    # temp_digits = datasets.load_digits()
    # digits = utils.resample(train_set, random_state=0)
    # temp_answers = utils.resample(ans_train, random_state=0)

    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # num_of_training_vectors = 29400
    # answers, answers_to_test, validation_answers = temp_answers, temp_answers[num_of_training_vectors:num_of_training_vectors+6300], temp_answers[num_of_training_vectors+6300:]
    # training_set, testing_set, validation_set = digits, test_set, digits[num_of_training_vectors+6300:]
    # epoch = 75
    # network = Network(target_values, training_set, answers, epoch, testing_set,
    #                   answers_to_test, validation_set, validation_answers)
    # network.learn_run()
    # network.report_results(network.run_unseen())
    # network.report_results(network.run_unseen(True), True)
    

# look at round where last backprop runs.  Maybe peel off one iteration?
# Get over it and append bias to forward pass, but not backward pass 
    ###########
    # visualization(train_set[10], ans_train[10])
    # visualization(train_set[11], ans_train[11])
    # visualization(train_set[12], ans_train[12])
    epochs = 100
    layers = 3
    neuron_count = [100, 100, 10]
    network = Network(layers, neuron_count, train_set[0])
    network.train(train_set, ans_train, epochs)
    
    guess_list = network.run_unseen(test_set)
    with open('digits.txt', 'w') as d:
        for elem in guess_list:
            d.write(str(elem)+'\n')

    # guess_list = network.run_unseen(testing_set)
    # network.report_results(guess_list, answers_to_test)
    # valid_list = network.run_unseen(validation_set)
    # network.report_results(valid_list, validation_answers)
