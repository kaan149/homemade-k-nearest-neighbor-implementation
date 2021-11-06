import numpy as np
import pandas as pd
from math import sqrt


# Euclidian distance calculates the span of two different points
# It is the base of k nearest neighbor algorithm
# It is starting point to clustering data with their alike
def euclidian_distance(sample1, sample2, length):
    distance = 0
    for i in range(length):
        distance += pow(sample1[i] - sample2[i], 2)

    return sqrt(distance)


# This function is used for finding neighbors. That get number of k minimum distances between one point and others
# It assigns a type with respect to neighbors of a point
def k_nearest_neighbor(x_train, x_test, y_train, k):
    predictions = []
    for i in range(len(x_test)):
        predictions.append(most_common(get_types_from_y_train(find_neighbors(x_train, x_test, i, k), y_train)))

    return predictions


# This function returns type of k minimum distance neighbor of a point by taking parameters as train and test data
# It also gets index as a parameter because it calculates every test element singularly
# It gets k as a parameter because it has to restrict minimum list as k
def find_neighbors(train, test, index, k):
    values = []
    indices = []
    for i in range(len(train)):
        val = euclidian_distance(train[i], test[index], 8)
        values.append(val)

    minimums = get_k_min_values(values, k)

    for val in minimums:
        indices.append(values.index(val))

    return indices


# It returns types of element which is as a neighbor
# It gets parameters as index of neighbor and find their types from y_train data
def get_types_from_y_train(indices, train):
    types = []
    for i in indices:
        types.append(train[i])

    return types


def most_common(lst):
    return max(set(lst), key=lst.count)


def get_k_min_values(values, k):
    return sorted(values)[:k]


# for weighted knn
def get_k_max_values(values, k):
    return sorted(values, reverse=True)[:k]


def calculate_accuracy(bool_arr):
    return 100 * (sum(bool_arr) / len(bool_arr))


# This function controls whether every part of data contains every type of data or not
def control_data_distribution(full_array, raw_types):
    constant = 0
    for sub_array in full_array:
        types = np.array([i[-1] for i in sub_array], dtype=np.int64)
        if not np.array_equiv(np.unique(types), raw_types):
            constant += 1
        else:
            pass

    if constant > 0:
        return True
    else:
        return False


def split_data(data):
    np.random.shuffle(data)
    return np.array_split(data, 5)


# It returns both information and their types separately
# It is for part 2 because we do not need to make a classification and distribute data equally (as type)
def divide_data(data):
    full_array = split_data(data)
    return [[j[:-1] for j in i] for i in full_array], [[j[-1] for j in i] for i in full_array]


# It divides data by controlling data distribution and returns material
def divide_data_equally(data, raw_types):
    full_array = split_data(data)
    while control_data_distribution(full_array, raw_types):
        divide_data_equally(data, raw_types)

    return [[j[:-1] for j in i] for i in full_array], [[j[-1] for j in i] for i in full_array]


# Normalization is for transforming column values between 0 and 1
# Thus we turn the data's coefficients more proper
def normalization(data, index):
    column = []
    for sub in data:
        column.append(sub[index])

    normalized = []
    max_val = max(column)
    min_val = min(column)
    for val in column:
        new_val = (val - min_val) / (max_val - min_val)
        normalized.append(new_val)

    return normalized


# Normalize every column
def normalize_data(data):
    new_source = []
    for i in range(len(data[0])):
        new_source.append(normalization(data, index=i))

    return new_source


def get_weight(dist):
    try:
        return 1 / dist
    except ZeroDivisionError:
        return 0


def calculate_mae(predicted, test):
    n = len(predicted)
    mae = (1 / n) * sum(abs(predicted[i] - test[i]) for i in range(n))
    return mae


# Weighted knn is for take the amount of proximity into account
# With using weighted knn we can make proper predictions
def weighted_k_nearest_neighbor(x_train, x_test, y_train, k):
    predictions = []
    for i in range(len(x_test)):
        predictions.append(create_dict(find_weighted_neighbors(x_train, x_test, i, k, y_train))[0][0])

    return predictions


# To find weighted neighbors we need to get k closest neighbor and calculate a value how a neighbor
# effects the point one by one
def find_weighted_neighbors(x_train, x_test, index, k, y_train):
    result = []
    values = []
    for i in range(len(x_train)):
        val = get_weight(euclidian_distance(x_train[i], x_test[index], 8))
        data_type = y_train[i]
        result.append([val, data_type])
        values.append(val)

    return get_k_max_values(result, k)


# Dictionary is for creating a data which neighbor type is close and what is the value of it
# For Instance k is 5 and type 1 neighbor is 3 times then the type is key and sum of weigh is value.
def create_dict(result):
    source = {}
    for i in result:
        if i[1] not in source:
            source[i[1]] = i[0]
        else:
            source[i[1]] += i[0]

    return list(sorted(source.items(), reverse=True, key=lambda x: x[1]))


def part2(x_source, y_source, columns, k):
    print("K VALUE IS: ", k)
    for i in range(len(x_source)):
        x_train = []
        y_train = []
        x_test = x_source[i]
        y_test = y_source[i]

        except_i = [j for j in range(len(x_source)) if j != i]

        for m in except_i:
            x_train.extend(x_source[m])
            y_train.extend(y_source[m])

        predicted = k_nearest_neighbor(x_train, x_test, y_train, k)

        mae = calculate_mae(predicted, y_test)
        print("mae plain = ", mae)

        normalized_x_train = list(map(list, zip(*normalize_data(x_train))))
        normalized_x_test = list(map(list, zip(*normalize_data(x_test))))

        train_df = pd.DataFrame(normalized_x_train[:], columns=columns)
        test_df = pd.DataFrame(normalized_x_test[:], columns=columns)
        predicted2 = k_nearest_neighbor(train_df.values, test_df.values, y_train, k)
        mae2 = calculate_mae(predicted2, y_test)
        print("mae2 normalized = ", mae2)

        predicted_weighted = weighted_k_nearest_neighbor(x_train, x_test, y_train, k)
        mae3 = calculate_mae(predicted_weighted, y_test)
        print("mae3 weighted but not normalized = ", mae3)

        predicted_weighted2 = weighted_k_nearest_neighbor(train_df.values, test_df.values, y_train, k)
        mae4 = calculate_mae(predicted_weighted2, y_test)
        print("mae4 both weighted and normalized = ", mae4)
        print("************************************************")


# We divided data 5 parts and we assign each of them as test data
# For part 1
def cross_validation(x_source, y_source, columns, k):
    print("K VALUE IS: ", k)
    for i in range(len(x_source)):
        x_train = []
        y_train = []
        x_test = x_source[i]
        y_test = y_source[i]

        except_i = [j for j in range(len(x_source)) if j != i]

        for m in except_i:
            x_train.extend(x_source[m])
            y_train.extend(y_source[m])

        predicted = k_nearest_neighbor(x_train, x_test, y_train, k)
        output = [i == j for i, j in zip(predicted, y_test)]
        print(calculate_accuracy(output))

        normalized_x_train = list(map(list, zip(*normalize_data(x_train))))
        normalized_x_test = list(map(list, zip(*normalize_data(x_test))))

        train_df = pd.DataFrame(normalized_x_train[:], columns=columns)
        test_df = pd.DataFrame(normalized_x_test[:], columns=columns)
        predicted2 = k_nearest_neighbor(train_df.values, test_df.values, y_train, k)
        output2 = [i == j for i, j in zip(predicted2, y_test)]
        print(calculate_accuracy(output2))

        predicted_weighted = weighted_k_nearest_neighbor(x_train, x_test, y_train, k)
        output_weighted = [i == j for i, j in zip(predicted_weighted, y_test)]
        print("WEIGHTED BUT NOT NORMALIZED", calculate_accuracy(output_weighted))

        predicted_weighted2 = weighted_k_nearest_neighbor(train_df.values, test_df.values, y_train, k)
        output_weighted2 = [i == j for i, j in zip(predicted_weighted2, y_test)]
        print("BOTH WEIGHTED AND NORMALIZED", calculate_accuracy(output_weighted2))
        print("*******************************************")


def main():
    df = pd.read_csv("glass.csv")
    x = df.values
    y = df["Type"].values
    columns = list(df.columns)
    columns.pop()
    x_source, y_source = divide_data_equally(x, np.unique(y))
    print("PART 1 DATA --- GLASS")

    # for testing k = 1, 3, 5, 7, 9
    for i in range(1, 10, 2):
        cross_validation(x_source, y_source, columns, k=i)

    df2 = pd.read_csv("Concrete_Data_Yeh.csv")
    x2 = df2.values
    columns2 = list(df2.columns)
    columns2.pop()
    x2_source, y2_source = divide_data(x2)

    print("PART 2 DATA --- CONCRETENESS")
    for i in range(1, 10, 2):
        part2(x2_source, y2_source, columns2, i)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
