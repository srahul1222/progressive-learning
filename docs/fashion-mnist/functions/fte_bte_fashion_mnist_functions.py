import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from proglearn.forest import LifelongClassificationForest


def run_parallel_exp(data_x, data_y, num_points_per_task, ntrees=30, slot=0, shift=1):

    train_x, train_y, test_x, test_y = cross_val_data(
        data_x, data_y, num_points_per_task
    )
    
    # Reshape the data 
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    df = label_shuffle_experiment(
        train_x,
        train_y,
        test_x,
        test_y,
        ntrees,
        shift,
        slot,
        num_points_per_task,
        acorn=12345,
    )
    return df


def cross_val_data(data_x, data_y, num_points_per_task, total_cls=10):
    # Creates copies of both data_x and data_y so that they can be modified without affecting the original sets
    x = data_x.copy()
    y = data_y.copy()
    # Creates a sorted array of arrays that each contain the indices at which each unique element of data_y can be found
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    for i in range(total_cls):
        # Chooses the i'th array within the larger idx array
        indx = idx[i]
        # The elements of indx are randomly shuffled
        random.shuffle(indx)

        if i == 0:
            # num_points_per_task training data points per task
            train_x = x[indx[0:num_points_per_task], :]
            train_y = y[indx[0:num_points_per_task]]

            # 1000 testing data points per task
            test_x = x[indx[num_points_per_task:7000], :]
            test_y = y[indx[num_points_per_task:7000]]
        else:
            # num_points_per_task training data points per task
            train_x = np.concatenate((train_x, x[indx[0:num_points_per_task], :]), axis=0)
            train_y = np.concatenate((train_y, y[indx[0:num_points_per_task]]), axis=0)

            # 1000 testing data points per task
            test_x = np.concatenate((test_x, x[indx[num_points_per_task:7000], :]), axis=0)
            test_y = np.concatenate((test_y, y[indx[num_points_per_task:7000]]), axis=0)

    return train_x, train_y, test_x, test_y



def label_shuffle_experiment(
    train_x,
    train_y,
    test_x,
    test_y,
    ntrees,
    shift,
    slot,
    num_points_per_task,
    acorn=None,
):

    # We initialize lists to store the results
    df = pd.DataFrame()
    shifts = []
    accuracies_across_tasks = []

    # Declare the progressive learner model (L2F)
    learner = LifelongClassificationForest()

    for task_num in range(5):
        print("Starting Task {} For Fold {} For Slot {}".format(task_num, shift, slot))
        if acorn is not None:
            np.random.seed(acorn)

        # If task number is 0, add task. Else, add a transformer for the task
        if task_num == 0:
            rand_idx = np.random.randint(0, 12000, 60)
            #print("Added task")
            learner.add_task(
                X = train_x[rand_idx, :],
                y = train_y[rand_idx],
#                 X=train_x[
#                     (task_num * 6000
#                     + slot * num_points_per_task) : (task_num * 6000
#                     + (slot + 1) * num_points_per_task), :
#                 ],
#                 y=train_y[
#                     (task_num * 6000
#                     + slot * num_points_per_task) : (task_num * 6000
#                     + (slot + 1) * num_points_per_task)
#                 ],
                task_id=0,
            )
        else:
            rand_idx = np.random.randint(0, 12000, 60)
            #print("Adding transformer")
            learner.add_transformer(
                X = train_x[((task_num * 12000) + rand_idx), :],
                y = train_y[((task_num * 12000) + rand_idx)],
#                 X=train_x[
#                     (task_num * 6000
#                     + slot * num_points_per_task) : (task_num * 6000
#                     + (slot + 1) * num_points_per_task), :
#                 ],
#                 y=train_y[
#                     (task_num * 6000
#                     + slot * num_points_per_task) : (task_num * 6000
#                     + (slot + 1) * num_points_per_task)
#                 ],
            )

        # Make a prediction on task 0 using the trained learner on test data
        llf_task = learner.predict(test_x[0:1000], task_id=0)
        #print(llf_task)

        # Calculate the accuracy of the task 0 predictions
        acc = np.mean(llf_task == test_y[0:1000])
        accuracies_across_tasks.append(acc)
        shifts.append(shift)

        print("Accuracy Across Tasks: {}".format(accuracies_across_tasks))

    df["data_fold"] = shifts
    df["task"] = range(1, 6)
    df["task_1_accuracy"] = accuracies_across_tasks

    return df


def get_bte(err):
    bte = []

    for i in range(10):
        bte.append(err[0] / err[i])

    return bte


def calc_bte(df_list, slots, shifts):
    shifts = shifts - 1
    reps = slots * shifts
    btes = np.zeros((1, 5), dtype=float)

    bte_tmp = [[] for _ in range(reps)]

    count = 0
    for shift in range(shifts):
        for slot in range(slots):

            # Get the dataframe containing the accuracies for the given shift and slot
            multitask_df = df_list[slot + shift * slots]
            err = []

            for ii in range(10):
                err.extend(
                    1
                    - np.array(
                        multitask_df[multitask_df["task"] == ii + 1]["task_1_accuracy"]
                    )
                )
            # Calculate the bte from task 1 error
            bte = get_bte(err)

            bte_tmp[count].extend(bte)
            count += 1

        # Calculate the mean backwards transfer efficiency
        btes[0] = np.mean(bte_tmp, axis=0)
    return btes


def plot_bte(btes):
    # Initialize the plot and color
    clr = ["#00008B"]
    c = sns.color_palette(clr, n_colors=len(clr))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot the results
    ax.plot(np.arange(1, 6), btes, c=c[0], label="L2F", linewidth=3)

    # Format the plot, and show result
    plt.ylim(0.92, 1.08)
    plt.xlim(1, 5)
    ax.set_yticks([0.92, 0.96, 1, 1.04, 1.08])
    ax.yaxis.set_ticks([0.92, 0.96, 1, 1.04, 1.08])
    ax.set_xticks(np.arange(1, 6))
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Number of tasks seen", fontsize=24)
    ax.set_ylabel("Backward Transfer Efficiency", fontsize=24)
    ax.set_title("BTE for Fashion-MNIST", fontsize=24)
    ax.hlines(1, 1, 10, colors="grey", linestyles="dashed", linewidth=1.5)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    plt.show()
