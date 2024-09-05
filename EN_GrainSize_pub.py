'''
This script will be for the use of an Elastic Net model to parse out which factors 
matter most in predicting electrolyte grain size from processing parameters.
Script written with the help of ChatGPT
'''

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import matplotlib as mpl
import matplotlib.transforms as mtrans
import cmasher as cmr
import optuna
import optuna.visualization as ov
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

# - Font Stuff
from matplotlib import font_manager
SSP_r = '/Library/Fonts/SourceSansPro-Regular.ttf'
SSP_b = '/Library/Fonts/SourceSansPro-Bold.ttf'
font_manager.fontManager.addfont(SSP_b)
font_manager.fontManager.addfont(SSP_r)
plt.rcParams['font.family'] = 'Source Sans Pro'


' ------ Lines to Edit'
model = 'GrainSize_model' #'GrainSize_SprayConditions' 'GrainSize_model'
# - Hyperparameters
if model == 'GrainSize_model':
    alpha = 0.01521 # 0.01521
    l1_ratio = 0.15534 # 0.15534 
elif model == 'GrainSize_SprayConditions':
    alpha = 0.424521 # 0.42452
    l1_ratio = 0.01 # 0.01 
tune_hyperparameters = False

# - Hyperparameter tuning
trials = 500
a_min = 0.0001
a_max = 1 # 100
il_min = 0.01
il_max = 1

# - Visualization
datapoints = True
print_coefs = False
plot_coefs = False
plot_all_residuals = False
plot_coefficients_residuals = True
plot_learning_curve = True

if tune_hyperparameters == True:
    print_coefs = False
    plot_coefs = False
coefs_to_print = 55
coefs_to_plot = 25
coef_to_latex = False
print_drop_batch = True
plot_drop_batch = True

# - Saving:
save_coef_plot = None # Change none to the path to where you would like to save the figure
save_resid_plot = None # 
save_coef_resid_plot = None # '
save_learning_curve_plot = None #


' ----- End Lines to Edit'
gs_data = 'path_to_data' #
df = pd.read_excel(gs_data,model) 

df = df.reindex(sorted(df.columns), axis=1)

df = df.copy()
df.rename(columns={'Sintering neighbor': 'SN'}, errors='raise',  inplace=True)
df.rename(columns={'Absolute humidity (g/m^3)': 'Absolute humidity (g/m$^3$)'}, errors='raise',  inplace=True)
df['SN'] = df['SN'].str.replace('BaCO3', 'BaCO$_3$', regex=False)

# # Count the occurrences of each category/word in the specified column
# category_counts = df['SN'].value_counts()
# print(category_counts)

' ------- Z-score standarization of all numerical data'
# Initialize the StandardScaler
scaler = StandardScaler()

# Identify and standardize numerical columns
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column].dtype):
        df[column] = scaler.fit_transform(df[[column]])
        
' ------- Encoding Catagorical columns'
object_cols = df.select_dtypes(include=['object']).columns.tolist() # Get a list of all the object (categorical) columns
for col in object_cols: #  Iterate through the object columns and apply one-hot encoding
    one_hot = pd.get_dummies(df[col], prefix=col, drop_first=False)  # drop_first=True to avoid multicollinearity (Not necesseary for EN models)
    df = pd.concat([df, one_hot], axis=1)
    df.drop(col, axis=1, inplace=True)  # Drop the original categorical column

df = df.dropna() # Dropping all N/A values

if datapoints == True:
    print('Number of datapoints:', len(df['Electrolyte grain size (μm)']))

' ----- Elastic Net Model'
target = 'Electrolyte grain size (μm)' # Electrolyte grain size (μm) Shrinkage

# Split the data into features (X) and target (y)
X = df.drop(target, axis=1)  # Replace 'target_column_name' with the actual target column name
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an Elastic Net model
alpha = alpha # 
l1_ratio = l1_ratio # 
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)  # You can adjust alpha and l1_ratio

elastic_net.fit(X_train, y_train) # Fit the model to the training data

y_pred = elastic_net.predict(X_test) # Use the trained Elastic Net model to make predictions on the test data:

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.5f}")

' ----- Automated Hyperparameter Tuning'
if tune_hyperparameters == True:
    def objective(trial):
        alpha = trial.suggest_float('alpha', a_min, a_max) 
        l1_ratio = trial.suggest_float('l1_ratio', il_min, il_max)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        return scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)  # You can adjust the number of trials

    best_hyperparameters = study.best_params # Get the best hyperparameters
    print('Suggested alpha:' , round(best_hyperparameters['alpha'],5))
    print('Suggested Il_ratio:' , round(best_hyperparameters['l1_ratio'],5))
    print(f"Mean Squared Error of last hyperparameters: {mse:.5f}")

    slice = optuna.visualization.plot_slice(study, params=['alpha', 'l1_ratio'])
    slice.show()

' ------- Printing the Coefficent list'
if print_coefs==True:
    feature_coefficients = elastic_net.coef_ # Get feature coefficients from the Elastic Net model
    coef_df = pd.DataFrame({'Feature': df.columns.drop(target), 'Coefficient': feature_coefficients}) # Create a DataFrame to associate feature names with coefficients

    coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs() # Sort the DataFrame by the absolute value of coefficients in descending order
    coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)

    if print_drop_batch == True:
        # Dropping Electrolyte spray batch:
        substring_to_drop = 'Electrolyte spray batch'
        coef_df = coef_df[~coef_df['Feature'].str.startswith(substring_to_drop)]
        ss2d = 'High temp sinter date'
        coef_df = coef_df[~coef_df['Feature'].str.startswith(ss2d)]

    # Display the top features by coefficient magnitude
    top_features = coef_df[['Feature', 'Coefficient']].head(coefs_to_print).reset_index(drop=True)  # Change the number as needed
    print(top_features)

    # Printing a latex table of the df:
    if coef_to_latex == True:
        print(coef_df[['Feature', 'Coefficient']].to_latex(index=False))

' ------- Plotting and visualzing the coefficients'
if plot_coefs == True:
    # Get feature coefficients from the model
    feature_coefficients = elastic_net.coef_ # Get feature coefficients from the Elastic Net model
    coef_df = pd.DataFrame({'Feature': df.columns.drop(target), 'Coefficient': feature_coefficients}) # Create a DataFrame to associate feature names with coefficients

    coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs() # Sort the DataFrame by the absolute value of coefficients in descending order
    coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)

    if plot_drop_batch == True:
        # Dropping Electrolyte spray batch:
        substring_to_drop = 'Electrolyte spray batch'
        coef_df = coef_df[~coef_df['Feature'].str.startswith(substring_to_drop)]
        ss2d = 'High temp sinter date'
        coef_df = coef_df[~coef_df['Feature'].str.startswith(ss2d)]

    # Data formatting for plotting
    # coef_df['Feature'] = coef_df['Feature'].str.slice(0, 27)
    print('Length of coef_df:', len(coef_df))

    n_rows_to_keep = coefs_to_plot
    df =coef_df.iloc[:n_rows_to_keep]

    fig,ax = plt.subplots()

    barplot = sns.barplot(data=df,x='Feature',y='Coefficient')

    # --- formatting and displaying
    ax.set_xlabel('Feature',size='x-large')
    ax.set_ylabel('Coefficient (a.u.)',size='xx-large', labelpad=5)
    new_labels = [f'$\\mathbf{{{i+1}}}$. {label}' for i, label in enumerate(df['Feature'])] # Number every entry
    if coefs_to_plot > 15:
        ax.set_xticklabels(new_labels, rotation=50, ha='right',size='small')
    else: 
        ax.set_xticklabels(new_labels, rotation=50, ha='right',size='large')
    ax.set(xlabel=None)
    ax.tick_params(axis='y', which='major', labelsize='large')
    ax.tick_params(axis='x', which='major', pad=0)

    # --- Colors
    if model == 'GrainSize_model':
        cmap = cmr.tropical_r
        colors = cmap(np.linspace(0.15, 1, 25))
    elif model == 'GrainSize_SprayConditions':
        cmap = cmr.rainforest_r
        light = 0.8 # 0.75 start = 2.5
        dark = 0.3 # 0.5
        rot = (1/(light-dark))*0.8 # 1.5
        colors = sns.cubehelix_palette(n_colors=25, start=2.3, rot=rot, gamma=1, hue=2, light=light, dark=dark, reverse=False, as_cmap=False)
            
    # ---- Add vertical dotted lines at every other x-value and change the barcolors
    ymin, ymax = ax.get_ylim()
    h_max = barplot.patches[0].get_height()
    for i, patch in enumerate(barplot.patches):
        # Get the x position for the center of each bar
        x = patch.get_x() + patch.get_width() / 2

        # Calculate ymax as a fraction of the y-axis range
        line_ymax = (patch.get_height() - ymin) / (ymax - ymin)
        height_of_bar = patch.get_height()
        if height_of_bar >= 0:
            line_ymax = (0 - ymin) / (ymax - ymin)

        # - Change the color of the bar
        patch.set_facecolor(colors[i])

        # Draw a line at every other x-value
        b_color = patch.get_facecolor()
        if i % 2 == 1:  # Change to i % 2 == 1 for every second bar starting with the second
            ax.axvline(x=x, color='grey', linestyle=(0, (4, 6.4)), linewidth=1, ymax=line_ymax)

    # Slight shift in xticklabels
    trans = mtrans.Affine2D().translate(10, 0) # (20,0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)

    sns.despine() # trim=True

    plt.tight_layout()

    if save_coef_plot is not None:
        fmat = save_coef_plot.split('.', 1)[-1]
        fig.savefig(save_coef_plot, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

' ---- Plotting residuals'
if plot_all_residuals == True:
    kf = KFold(n_splits=5, shuffle=True, random_state=42) # Initialize KFold
    residuals = [] # Arrays to store the residuals for each fold
    y_test_all = []  # To store all actual values
    y_pred_all = []  # To store all predicted values

    # Ensure X and y are numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Perform 5-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        elastic_net.fit(X_train, y_train)

        # Predict on the test set
        y_pred = elastic_net.predict(X_test)

        # Store actual and predicted values
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # Calculate residuals
        fold_residuals = y_test - y_pred
        residuals.extend(fold_residuals)

    # Convert residuals to a numpy array for easy plotting
    residuals = np.array(residuals)

    # Calculate R^2 from residuals
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_test_all - np.mean(y_test_all))**2)
    r2 = 1 - (ss_res / ss_tot)
    print(f'R^2: {r2}')

    # --- Plotting
    ssp_font = {'fontname':'Source Sans Pro'}
    plt.figure(figsize=(12, 6))
    subfig_label_fontsize = 34
    label_fs = 20 # 'xx-large'
    tick_fs = label_fs * 0.85 # 'x-large

    plt.subplot(1, 3, 1)

    plt.scatter(y_test_all, y_pred_all, alpha=0.5)  # Create a scatterplot of actual vs. predicted values
    plt.xlabel("Actual Values",fontsize=label_fs)
    plt.ylabel("Model-predicted values",fontsize=label_fs)
    plt.title("Actual vs. predicted values",fontsize=label_fs)
    plt.figtext(0.03,1.0, 'a', fontsize=subfig_label_fontsize, ha='left',va='top',weight='bold',**ssp_font)
    min_val = min(min(y_test_all), min(y_pred_all))
    max_val = max(max(y_test_all), max(y_pred_all))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Residuals vs. Predicted Values
    plt.subplot(1, 3, 2)

    plt.scatter(y_pred_all, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Model-predicted values",fontsize=label_fs)
    plt.ylabel("Residuals",fontsize=label_fs)
    plt.title("Residuals vs. predicted values",fontsize=label_fs)
    plt.figtext(0.35,1.0, 'b', fontsize=subfig_label_fontsize, ha='left',va='top',weight='bold',**ssp_font)
    plt.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Residuals Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True,stat="density")
    sns.kdeplot(residuals,color='r')
    plt.xlabel("Residuals",fontsize=label_fs)
    plt.ylabel("Frequency",fontsize=label_fs)
    plt.title("Residuals distribution",fontsize=label_fs)
    plt.figtext(0.70,1.0, 'c', fontsize=subfig_label_fontsize, ha='left',va='top',weight='bold',**ssp_font)
    plt.tick_params(axis='both', which='major', labelsize=tick_fs)

    plt.tight_layout()
        
    if save_resid_plot is not None:
        fmat = save_resid_plot.split('.', 1)[-1]
        plt.savefig(save_resid_plot, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

' ---- Plotting coefficients and residuals'
if plot_coefficients_residuals == True:
    # ----- Setting up plot
    fig = plt.figure(figsize=(9, 6.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.3, 1])
    
    # - Formatting
    subfig_label_fontsize = 26
    label_fs = 18 # 'xx-large'
    tick_fs = label_fs * 0.85 # 'x-large
    parameter_fs = 10
    subfig_label_x = -0.1
    subfig_label_y = 0.95
    ssp_font = {'fontname':'Source Sans Pro'}

    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[1, 0:1])
    ax3 = fig.add_subplot(gs[1, 1:2])
    ax4 = fig.add_subplot(gs[1, 2:3])

    ' ======= Calculating and plotting coefficients '
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_net.fit(X_train, y_train)

    # Get feature coefficients from the model
    feature_coefficients = elastic_net.coef_ # Get feature coefficients from the Elastic Net model
    coef_df = pd.DataFrame({'Feature': df.columns.drop(target), 'Coefficient': feature_coefficients}) # Create a DataFrame to associate feature names with coefficients

    coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs() # Sort the DataFrame by the absolute value of coefficients in descending order
    coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)
    
    if plot_drop_batch == True:
        ss2d = 'Sinter_Date'
        coef_df = coef_df[~coef_df['Feature'].str.startswith(ss2d)]

    # Data formatting for plotting
    print(' Length of coef_df:', len(coef_df))
    n_rows_to_keep = coefs_to_plot
    df =coef_df.iloc[:n_rows_to_keep]

    # - Setting up the colors:
    df['Overall_Category'] = df['Feature'].apply(lambda x: x.split('_')[0] if '_' in x else x)     # Extract overall categories
    unique_categories = df['Overall_Category'].unique() # Get unique overall categories
    category_palette = sns.color_palette('husl', n_colors=len(unique_categories))
    pal1 = sns.color_palette('tab10')
    pal2 = sns.color_palette('Dark2_r')
    pal3 = sns.color_palette('Set1_r')
    pal_tot = pal1 + pal2 + pal3
    palette_map = {category: color for category, color in zip(unique_categories, pal_tot)}
    df['Color'] = df['Overall_Category'].map(palette_map)

    # Normalize the absolute y-values (coefficients) to a range of 0 to 1.0
    norm = plt.Normalize(0, df['Coefficient'].abs().max())  # Normalize from 0 to max absolute coefficient value

    # Adjust the palette based on the lightness
    def adjust_lightness(color, lightness_factor):
        c = mcolors.to_rgb(color)
        adjusted_color = [lightness_factor * channel + (1 - lightness_factor) for channel in c]
        return adjusted_color

    barplot = sns.barplot(data=df,x='Feature',y='Coefficient',ax=ax1,
                          hue='Feature',legend=False)
    
    # Manually set colors for each bar
    for i, patch in enumerate(barplot.patches):
        overall_category = df['Overall_Category'].iloc[i]
        base_color = palette_map[overall_category]
        min_lightness = 0.7 # Adjust min-lightness of the bars
        lightness_factor = norm(abs(df['Coefficient'].iloc[i])) * (1-min_lightness) + min_lightness  # Scale to 0.75 - 1.0
        adjusted_color = adjust_lightness(base_color, lightness_factor)
        patch.set_color(adjusted_color)
        patch.set_edgecolor('black')

    # --- formatting and displaying
    ax1.set_xlabel('Feature',size=label_fs)
    ax1.set_ylabel('Coefficient (a.u.)',size=label_fs, labelpad=10)
    new_labels = [f'$\\mathbf{{{i+1}}}$. {label}' for i, label in enumerate(df['Feature'])] # Number every entry
    
    if coefs_to_plot < 15:
        parameter_fs = 15
    
    ax1.set_xticklabels(new_labels, rotation=50, ha='right',size=parameter_fs)

    ax1.set(xlabel=None)
    ax1.tick_params(axis='y', which='major', labelsize=tick_fs)
    ax1.tick_params(axis='x', which='major', pad=0)

    ax1.text(-0.02,subfig_label_y, 'a', fontsize=subfig_label_fontsize, ha='right',va='bottom',weight='bold',
             transform=ax1.transAxes,**ssp_font)

    # ---- Add vertical dotted lines at every other x-value
    ymin, ymax = barplot.get_ylim()
    for i, patch in enumerate(barplot.patches):
        x = patch.get_x() + patch.get_width() / 2

        # Calculate ymax as a fraction of the y-axis range
        line_ymax = (patch.get_height() - ymin) / (ymax - ymin)
        height_of_bar = patch.get_height()
        
        if height_of_bar >= 0:
            line_ymax = (0 - ymin) / (ymax - ymin)

        # Draw a line at every other x-value
        if i % 2 == 1:  # Change to i % 2 == 1 for every second bar starting with the second
            ax1.axvline(x=x, color='grey', linestyle=(0, (4, 6.4)), linewidth=1, ymax=line_ymax)

    # Slight shift in xticklabels
    trans = mtrans.Affine2D().translate(20, 0)
    for t in ax1.get_xticklabels():
        t.set_transform(t.get_transform()+trans)

    sns.despine() # trim=True

    ' ======= Calculating residuals for all folds of cross validation '
    kf = KFold(n_splits=5, shuffle=True, random_state=42) # Initialize KFold
    residuals = [] # Arrays to store the residuals for each fold
    y_test_all = []  # To store all actual values
    y_pred_all = []  # To store all predicted values

    # Ensure X and y are numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Perform 5-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        elastic_net.fit(X_train, y_train)

        # Predict on the test set
        y_pred = elastic_net.predict(X_test)

        # Store actual and predicted values
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # Calculate residuals
        fold_residuals = y_test - y_pred
        residuals.extend(fold_residuals)

    # Convert lists to numpy arrays for easy plotting
    residuals = np.array(residuals)
    y_test_all = np.array(y_test_all)
    y_pred_all = np.array(y_pred_all)    

    # Calculate R^2 from residuals
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_test_all - np.mean(y_test_all))**2)
    r2 = 1 - (ss_res / ss_tot)
    print(f'R^2: {r2}')
    r2_str = r'R$^2$ = ' + f'{r2:.2f}'

    # --- Plotting
    ax2.scatter(y_test_all, y_pred_all, alpha=0.5)  # Create a scatterplot of actual vs. predicted values
    ax2.set_xlabel(r"Measured/$\sigma$",fontsize=label_fs)
    ax2.set_ylabel(r"Predicted/$\sigma$",fontsize=label_fs)
    ax2.text(subfig_label_x,subfig_label_y, 'b', fontsize=subfig_label_fontsize, ha='right',va='bottom',weight='bold',
             transform=ax2.transAxes,**ssp_font)
    ax2.text(0.02,1, r2_str, fontsize=tick_fs, ha='left',va='top',weight='bold',
             transform=ax2.transAxes,**ssp_font)
    min_val = min(min(y_test_all), min(y_pred_all))
    max_val = max(max(y_test_all), max(y_pred_all))
    ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Residuals vs. Predicted Values
    ax3.scatter(y_pred_all, residuals, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel(r"Predicted values/$\sigma$",fontsize=label_fs)
    ax3.set_ylabel("Residuals",fontsize=label_fs)
    ax3.text(subfig_label_x,subfig_label_y, 'c', fontsize=subfig_label_fontsize, ha='right',va='bottom',weight='bold',
             transform=ax3.transAxes,**ssp_font)
    ax3.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Residuals Distribution
    sns.histplot(residuals, kde=True,stat="density")
    sns.kdeplot(residuals,color='r')
    ax4.set_xlabel("Residuals",fontsize=label_fs)
    ax4.set_ylabel("Frequency",fontsize=label_fs)
    ax4.text(subfig_label_x,subfig_label_y, 'd', fontsize=subfig_label_fontsize, ha='right',va='bottom',weight='bold',
             transform=ax4.transAxes,**ssp_font)
    ax4.tick_params(axis='both', which='major', labelsize=tick_fs)

    plt.tight_layout()

    if save_coef_resid_plot is not None:
        fmat = save_coef_resid_plot.split('.', 1)[-1]
        fig.savefig(save_coef_resid_plot, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

' ===== Learning Curve ===== '
if plot_learning_curve == True:
    # --- Initializing plot
    fig, ax = plt.subplots()
    
    # --- Calculating learning curves
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, validation_scores = learning_curve(
        elastic_net, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=kf, scoring='neg_mean_squared_error'
    )

    # --- Calculate mean and standard deviation for training and validation scores
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    ax.plot(train_sizes, validation_scores_mean, 'o-', color='green', label='Validation score',markersize=10,lw=2)
    ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score',markersize=10,lw=2)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.15, color='blue')
    ax.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.15, color='green')
    
    # - Formatting
    label_fs = 24 # 'xx-large'
    tick_fs = label_fs * 0.85 # 'x-large
    legend_fs = label_fs * 0.65

    ax.set_xlabel('Training set size',size=label_fs)
    ax.set_ylabel('Mean squared error (MSE)',size=label_fs)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.legend(fontsize=legend_fs)

    plt.tight_layout()

    if save_learning_curve_plot is not None:
        fmat = save_learning_curve_plot.split('.', 1)[-1]
        fig.savefig(save_learning_curve_plot, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()