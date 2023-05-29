import numpy as np
import matplotlib.pyplot as plt

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true,axis=1,keepdims=True)
    total_sum_squares = np.sum((y_true - y_mean)**2)
    residual_sum_squares = np.sum((y_true - y_pred)**2)
    r2 = 1 - (residual_sum_squares / total_sum_squares)
    return r2

def plot_Scalar_Range(x_test_dict,y_test_dict,y_predict_dict,angular_fixed,scalar_fixed):

    # The relation between Scalar-Range when angular is fixed at angular_fixed degree
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['lines.linewidth'] = 3
    # Set the figure size
    plt.figure(figsize=(10, 6))  # Adjust the width and height values as per your preference

    # Plotting the real values as blue dots
    for i, a in enumerate(angular_fixed):
        x_test_temp = x_test_dict["x_test" + str(i)]
        y_test_temp = y_test_dict["y_test" + str(i)]
        y_predict_temp = y_predict_dict["y_predict" + str(i)]
        m = y_test_temp.shape[1]
        accuracy = r2_score(y_test_temp[0, :].reshape((1, m)), y_predict_temp[0, :].reshape((1, m)))
        plt.scatter(x_test_temp[0, :], y_test_temp[0, :], color='blue')
        plt.plot(x_test_temp[0, :], y_predict_temp[0, :], color='red')
        plt.text(x_test_temp[0, -1], y_test_temp[0, -1], str(a) + "°", color='black', verticalalignment='bottom',
                 horizontalalignment='right')
        plt.text(x_test_temp[0, 0], y_test_temp[0, -1], f"({a}°)Accuracy: {accuracy:.5f}", color='black',
                 verticalalignment='top', horizontalalignment='left')
    # Add labels and title to the plot
    plt.xlabel('scalar[m/s]')
    plt.ylabel('Range [m]')
    plt.title('Comparison of Y_predict and Y_real in Scalar-Range')
    legend_entries = [plt.Line2D([0], [0], marker='o', color='blue', label='Real values'),
                      plt.Line2D([0], [0], marker='', color='red', label='Predicted values')]
    # Add the legend with custom entries
    plt.legend(handles=legend_entries, loc='upper center')
    plt.show()

def plot_Scalar_Height(x_test_dict,y_test_dict,y_predict_dict,angular_fixed,scalar_fixed):
    # The relation between Scalar-Height when angular is fixed at angular_fixed degree
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['lines.linewidth'] = 3
    # Set the figure size
    plt.figure(figsize=(10, 6))  # Adjust the width and height values as per your preference

    # Plotting the real values as blue dots
    for i, a in enumerate(angular_fixed):
        x_test_temp = x_test_dict["x_test" + str(i)]
        y_test_temp = y_test_dict["y_test" + str(i)]
        y_predict_temp = y_predict_dict["y_predict" + str(i)]
        m = y_test_temp.shape[1]
        accuracy = r2_score(y_test_temp[1, :].reshape((1, m)), y_predict_temp[1, :].reshape((1, m)))
        plt.scatter(x_test_temp[0, :], y_test_temp[1, :], color='blue')
        plt.plot(x_test_temp[0, :], y_predict_temp[1, :], color='red')
        plt.text(x_test_temp[0, -1], y_test_temp[1, -1], str(a) + "°", color='black', verticalalignment='bottom',
                 horizontalalignment='right')
        plt.text(x_test_temp[0, 0], y_test_temp[1, -1], f"({a}°)Accuracy: {accuracy:.5f}", color='black',
                 verticalalignment='top', horizontalalignment='left')
    # Add labels and title to the plot
    plt.xlabel('scalar[m/s]')
    plt.ylabel('Height [m]')
    plt.title('Comparison of Y_predict and Y_real in Scalar-Height')
    legend_entries = [plt.Line2D([0], [0], marker='o', color='blue', label='Real values'),
                      plt.Line2D([0], [0], marker='', color='red', label='Predicted values')]
    # Add the legend with custom entries
    plt.legend(handles=legend_entries, loc='upper center')
    plt.show()

def plot_Angular_Range(x_test_dict,y_test_dict,y_predict_dict,angular_fixed,scalar_fixed):
    # The relation between Angular-Range when range is fixed at scalar_fixed m/s
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['lines.linewidth'] = 3
    # Set the figure size
    plt.figure(figsize=(10, 6))  # Adjust the width and height values as per your preference
    offset = len(angular_fixed)
    # Plotting the real values as blue dots
    for i, s in enumerate(scalar_fixed):
        x_test_temp = x_test_dict["x_test" + str(i + offset)]
        y_test_temp = y_test_dict["y_test" + str(i + offset)]
        y_predict_temp = y_predict_dict["y_predict" + str(i + offset)]
        m = y_test_temp.shape[1]
        accuracy = r2_score(y_test_temp[0, :].reshape((1, m)), y_predict_temp[0, :].reshape((1, m)))
        plt.scatter(x_test_temp[1, :], y_test_temp[0, :], color='blue')
        plt.plot(x_test_temp[1, :], y_predict_temp[0, :], color='red')
        plt.text(x_test_temp[1, 500], y_test_temp[0, 500], f"{s} m/s(Accuracy: {accuracy:.5f})", color='black',
                 verticalalignment='top', horizontalalignment='left')
    # Add labels and title to the plot
    plt.xlabel('angular [°]')
    plt.ylabel('Range [m]')
    plt.title('Comparison of Y_predict and Y_real in Angular-Range')
    legend_entries = [plt.Line2D([0], [0], marker='o', color='blue', label='Real values'),
                      plt.Line2D([0], [0], marker='', color='red', label='Predicted values')]
    # Add the legend with custom entries
    plt.legend(handles=legend_entries)
    plt.show()

def plot_Angular_Height(x_test_dict,y_test_dict,y_predict_dict,angular_fixed,scalar_fixed):
    # The relation between Angular-Height when range is fixed at scalar_fixed m/s
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['lines.linewidth'] = 3
    # Set the figure size
    plt.figure(figsize=(10, 6))  # Adjust the width and height values as per your preference
    offset = len(angular_fixed)
    # Plotting the real values as blue dots
    for i, s in enumerate(scalar_fixed):
        x_test_temp = x_test_dict["x_test" + str(i + offset)]
        y_test_temp = y_test_dict["y_test" + str(i + offset)]
        y_predict_temp = y_predict_dict["y_predict" + str(i + offset)]
        m = y_test_temp.shape[1]
        accuracy = r2_score(y_test_temp[1, :].reshape((1, m)), y_predict_temp[1, :].reshape((1, m)))
        plt.scatter(x_test_temp[1, :], y_test_temp[1, :], color='blue')
        plt.plot(x_test_temp[1, :], y_predict_temp[1, :], color='red')
        plt.text(x_test_temp[1, -1], y_test_temp[1, -1], str(s) + "m/s", color='black', verticalalignment='bottom',
                 horizontalalignment='right')
        plt.text(x_test_temp[1, 0], y_test_temp[1, -1], f"({s} m/s)Accuracy: {accuracy:.5f}", color='black',
                 verticalalignment='top', horizontalalignment='left')
    # Add labels and title to the plot
    plt.xlabel('angular [°]')
    plt.ylabel('Height [m]')
    plt.title('Comparison of Y_predict and Y_real in Angular-Height')
    legend_entries = [plt.Line2D([0], [0], marker='o', color='blue', label='Real values'),
                      plt.Line2D([0], [0], marker='', color='red', label='Predicted values')]
    # Add the legend with custom entries
    plt.legend(handles=legend_entries, loc='upper center')
    plt.show()