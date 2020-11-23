import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from proglearn.forest import LifelongClassificationForest


def run_bte_exp(data_x, data_y, num_points_per_task, ntrees=30, slot=0):
    
    df_total = []
    
    for shift in range(10):
        train_x, train_y, test_x, test_y = cross_val_data(
            data_x, data_y, slot, shift
        )

        # Reshape the data 
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2] * train_x.shape[3])
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2] * test_x.shape[3])

        df = bte_experiment(
            train_x,
            train_y,
            test_x,
            test_y,
            ntrees,
            slot,
            shift,
            num_points_per_task,
            acorn=12345,
        )
        
        df_total.append(df)
        
    return df_total


def cross_val_data(data_x, data_y, slot, shift, total_cls=100):
    # Creates copies of both data_x and data_y so that they can be modified without affecting the original sets
    x = data_x.copy()
    y = data_y.copy()
    # Creates a sorted array of arrays that each contain the indices at which each unique element of data_y can be found
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    for i in range(total_cls):
        # Chooses the i'th array within the larger idx array
        indx = idx[i]
        ## The elements of indx are randomly shuffled
        #random.shuffle(indx)
        
        # 900 training data points per class
        tmp_x = np.concatenate((x[indx[0:(slot*100)], :], x[indx[((slot+1)*100):1000], :]), axis=0)
        tmp_y = np.concatenate((y[indx[0:(slot*100)]], y[indx[((slot+1)*100):1000]]), axis=0)
        
        if i == 0:
            # 90 training data points per class
            train_x = tmp_x[(shift*90):((shift+1)*90)]
            train_y = tmp_y[(shift*90):((shift+1)*90)]

            # 100 testing data points per class
            test_x = x[indx[(slot*100):((slot+1)*100)], :]
            test_y = y[indx[(slot*100):((slot+1)*100)]]
        else:
            # 900 training data points per class
            train_x = np.concatenate((train_x, tmp_x[(shift*90):((shift+1)*90)]), axis=0)
            train_y = np.concatenate((train_y, tmp_y[(shift*90):((shift+1)*90)]), axis=0)

            # 100 testing data points per class
            test_x = np.concatenate((test_x, x[indx[(slot*100):((slot+1)*100)], :]), axis=0)
            test_y = np.concatenate((test_y, y[indx[(slot*100):((slot+1)*100)]]), axis=0)

    return train_x, train_y, test_x, test_y



def bte_experiment(
    train_x,
    train_y,
    test_x,
    test_y,
    ntrees,
    slot,
    shift,
    num_points_per_task,
    acorn=None,
):

    # We initialize lists to store the results
    df = pd.DataFrame()
    accuracies_across_tasks = []

    # Declare the progressive learner model (L2F)
    learner = LifelongClassificationForest()

    for t_num in range(0,10):
        #task_num = 9 - t_num ==> for calculating FTE
        task_num = t_num
        accuracy_per_task = []
        print("Starting Task {} For Slot {} For Shift {}".format(task_num, slot, shift))
        if acorn is not None:
            np.random.seed(acorn)

        # If task number is 0, add task. Else, add a transformer for the task
        if task_num == 0:
            #rand_idx = np.random.randint(0, 9000, num_points_per_task)
            #print("Added task")
            learner.add_task(
                X = train_x[(task_num*900):((task_num + 1)*900)],
                y = train_y[(task_num*900):((task_num + 1)*900)],
                task_id=0,
            )
        else:
            #rand_idx = np.random.randint(0, 9000, num_points_per_task)
            #print("Adding transformer")
            learner.add_transformer(
                X = train_x[(task_num*900):((task_num + 1)*900)],
                y = train_y[(task_num*900):((task_num + 1)*900)],
            )

        # Make a prediction on task 0 using the trained learner on test data
        llf_task = learner.predict(test_x[0:1000, :], task_id=0)
        acc = np.mean(llf_task == test_y[0:1000])    
        accuracies_across_tasks.append(acc)
        print("Accuracy Across Tasks: {}".format(accuracies_across_tasks))

    df["task"] = range(1, 11)
    df["task_accuracy"] = accuracies_across_tasks

    return df


def get_bte(err):
    bte = []

    for i in range(10):
        bte.append(err[0] / err[i])

    return bte


def plot_bte(btes):
    # Initialize the plot and color
    clr = ["#00008B"]
    c = sns.color_palette(clr, n_colors=len(clr))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot the results
    ax.plot(np.arange(1, 11), btes, c=c[0], label="L2F", linewidth=3)

    # Format the plot, and show result
    plt.ylim(0.92, 1.16)
    plt.xlim(1, 10)
    ax.set_yticks([0.92, 0.96, 1, 1.04, 1.08, 1.12, 1.16])
    ax.yaxis.set_ticks([0.92, 0.96, 1, 1.04, 1.08, 1.12, 1.16])
    ax.set_xticks(np.arange(1, 11))
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Number of tasks seen", fontsize=24)
    ax.set_ylabel("Backward Transfer Efficiency", fontsize=24)
    ax.set_title("BTE for food-101", fontsize=24)
    ax.hlines(1, 1, 10, colors="grey", linestyles="dashed", linewidth=1.5)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    plt.show()
