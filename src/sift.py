import os
import cv2
import numpy as np
from scipy.cluster.vq import *
from sklearn import svm
from sklearn import preprocessing


def get_sub_dirs(root_dir):
    sub_dirs = []
    for root, dir_names, file_names in os.walk(root_dir):
        if dir_names:
            for name in dir_names:
                sub_dirs.append(name)
    return sub_dirs


def get_sift_descriptors_from_images(root_dir, sub_dirs, n):
    images = []
    labels = []
    number_of_images = 0
    sift = cv2.xfeatures2d.SIFT_create()
    for sub_dir in sub_dirs:
        for root, dir_names, file_names in os.walk(root_dir + sub_dir):
            number_of_images += len(file_names)
            # can restrict to n number of images in each class (sub dir)
            img_file_names = [file_name for ind, file_name in enumerate(file_names) if ind < n]

            # get sift descriptors from for images
            sift_descriptors = []
            for img_file_name in img_file_names:
                img = cv2.imread(root_dir + sub_dir + '/' + img_file_name)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kps, descs = sift.detectAndCompute(gray, None)
                sift_descriptors.append((img_file_name, descs))

                # append each label to the label list
                labels.append(int(sub_dir))

            # append tuple with sub dir (0, 1, 2, 3, 4, 5, 6) and a list of all the sift descriptors
            images.append((sub_dir, sift_descriptors))

    return images, labels, number_of_images


def stack_descriptors(sift_images):
    des_list = []
    for sub_dir, sift_descriptors in sift_images:
        des_list.append(sift_descriptors)
    des_list_flattened = [item for sublist in des_list for item in sublist]

    # Stack descriptor vectors (or matrices) vertically
    descriptors = des_list_flattened[0][1]
    for img_file_name, descriptor in des_list_flattened[1:]:
        if descriptor is not None:
            descriptors = np.concatenate((descriptors, descriptor), axis=0)

    return descriptors, des_list_flattened


def perform_k_means(descriptors, k):
    # will give k codewords (mean values of cluster centers)
    visual_dict, variance = kmeans(descriptors, k, 1)  # k-means with default values
    return visual_dict, variance


def compute_histogram_of_words(codebook, descriptors):
    # Alternative approach to computing the histogram of visual words
    code, dist = vq(descriptors, codebook)
    histogram_of_words, bin_edges = np.histogram(
        code,
        bins=range(codebook.shape[0] + 1),
        normed=True)
    return histogram_of_words


def calculate_bag_of_words_histogram(k, desc_list, codebook):
    img_features = np.zeros((len(desc_list), k), 'float32')
    for i in range(len(desc_list)):
        if desc_list[i][1] is not None:
            words, distance = vq(desc_list[i][1], codebook)
            for w in words:
                img_features[i][w] += 1
    return img_features


def svm_classify(x, y, test_ex):
    # uses one-vs-one method
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x, y)
    predicts = clf.predict(test_ex)
    return predicts


def svm_linear_classify(x, y, test_ex):
    # uses one-vs-the-rest method
    lin_clf = svm.LinearSVC()
    lin_clf.fit(x, y)
    predicts = lin_clf.predict(test_ex)
    return predicts


def test_accuracy(labels, predictions):
    correct_predictions = 0
    for i, prediction in enumerate(predictions):
        if prediction == labels[i]:
            correct_predictions += 1
    return (correct_predictions / len(labels)) * 100


def range_scaler(bag_of_words):
    # scale values to be between 0, 1
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_words = min_max_scaler.fit_transform(bag_of_words)
    return scaled_words


if __name__ == "__main__":
    number_of_imgs_from_each_class = 100

    tr_sub_dirs = get_sub_dirs('../data/img/train')
    # get sift descriptors for training images
    tr_images, tr_labels, n_tr_images = get_sift_descriptors_from_images(
        '../data/img/train/',
        tr_sub_dirs,
        number_of_imgs_from_each_class)

    # Stack all descriptors vertically
    stacked_descs, des_list = stack_descriptors(tr_images)
    print('number of training images: ', len(des_list))

    # Compute K-means group similar feature descriptors together
    k = 500
    code_book, var = perform_k_means(stacked_descs, k)
    print('training codebook of size: ', code_book.shape)

    # Create a bag of visual words histogram for each image, where each bin represent a codeword
    # which counts the number of words assigned to that codeword
    bag_of_visual_words = calculate_bag_of_words_histogram(k, des_list, code_book)
    print('histogram of visual words of size:', bag_of_visual_words.shape)

    # Get a sample of test images and repeat procedure to get feature descriptors, perform k-means and
    # create a bag of visual words histogram

    n_test_images_from_each_class = 10
    test_sub_dirs = get_sub_dirs('../data/img/test')

    # Get sift descriptors for test images
    test_images, test_labels, n_test_images = get_sift_descriptors_from_images(
        '../data/img/test/',
        test_sub_dirs,
        n_test_images_from_each_class)

    # Stack descriptors
    test_stacked_desc, test_des_list = stack_descriptors(test_images)
    print('number of test images: ', len(test_des_list))

    # Compute code words and write code book
    test_code_book, var = perform_k_means(test_stacked_desc, k)
    print('test codebook of size: ', test_code_book.shape)

    # Compute histogram of words
    test_bag_visual_words = calculate_bag_of_words_histogram(
        k,
        test_des_list,
        test_code_book)
    print('histogram of visual words of size:', test_bag_visual_words.shape)

    # SVM Classification
    # fit and predict using one-vs-one method
    predictions = svm_classify(
        range_scaler(bag_of_visual_words),
        tr_labels,
        range_scaler(test_bag_visual_words))

    # fit and predict using one-vs-the-rest method
    linear_predictions = svm_linear_classify(
        range_scaler(bag_of_visual_words),
        tr_labels,
        range_scaler(test_bag_visual_words))

    print('svm classification')
    print('test accuracy')

    accuracy = test_accuracy(test_labels, predictions)
    print('svm using one-vs-one')
    print(round(accuracy), '%')

    accuracy2 = test_accuracy(test_labels, linear_predictions)
    print('svm using one-vs-the-rest')
    print(round(accuracy2), '%')
