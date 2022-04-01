{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "q82tr780vc0K"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy import stats\n",
    "# import seaborn as sns\n",
    "import matplotlib.pylab as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "FOfkHBkZvjJ-",
    "outputId": "49ac4ce2-9845-4db0-b711-aa47e08d5fad"
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv('D:/research work/FailureMode_shear wall/input.csv')\n",
    "print('Shape of the data is ', data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "u7XX80d3vpNt",
    "outputId": "f4d50f55-cb8b-4f96-f0b0-8e58a3953e41"
   },
   "outputs": [],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "3RzylqXrvrdY",
    "outputId": "f8b96843-5cc2-4e2e-dfd1-6800482122b7"
   },
   "outputs": [],
   "source": [
    "data.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 377
    },
    "id": "GRCVcxLkvvMT",
    "outputId": "84575ba8-96d2-45f7-b681-41bbd1c69d94"
   },
   "outputs": [],
   "source": [
    "mypal= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA', '#FC05FB', '#FEAEFE', '#FCD2FC']\n",
    "\n",
    "plt.figure(figsize=(7, 5),facecolor='#F6F5F4')\n",
    "total = float(len(data))\n",
    "# ax = sns.countplot(x=data['Failure'], palette=mypal[1::1])\n",
    "ax = sns.countplot(x=data['Failure mode'])\n",
    "# ax.set_facecolor('#F6F5F4')\n",
    "\n",
    "for p in ax.patches:\n",
    "    \n",
    "    height = p.get_height()\n",
    "    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.1f} %'.format((height/total)*100), ha=\"center\")\n",
    "\n",
    "ax.set_title('Target variable distribution', fontsize=20, y=1.05)\n",
    "sns.set(font_scale=1.5)\n",
    "# plt.rc('font',family='Times New Roman',size=12)\n",
    "sns.despine(right=True)\n",
    "sns.despine(offset=5, trim=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.head(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Distribution of design and parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "nf = ['r', 'fck', 'n', 'fy', 'As', 'fyw', 'ρw', 'fsw', 'ρsw', 'Vf/Vs',  'Failure mode']\n",
    "data_ = data[nf]\n",
    "# sns.set(font_scale=1.5)\n",
    "#  plt.rc('font',family='Times New Roman',size=12)\n",
    "# plt.rcParams['figure.dpi'] = 1200\n",
    "# sns.set(style='whitegrid', rc={\"grid.linewidth\": 1},font='Times New Roman',font_scale=2)\n",
    "# sns.pairplot(data,hue=\"Failure mode\", diag_kind=\"hist\",markers=[\"o\", \"s\", \"D\"],palette=\"husl\")\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_feats = ['r', 'fck', 'n', 'fy', 'As', 'fyw', 'ρw', 'fsw', 'ρsw', 'Vf/Vs',  'Failure mode']\n",
    "df_ = data[num_feats]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "v7PFEIjVwMf5"
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC, LinearSVC, NuSVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n",
    "\n",
    "from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report\n",
    "from sklearn.metrics import recall_score, accuracy_score,roc_curve, auc\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import shap "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data.loc[:, data.columns != 'Failure mode']\n",
    "y=data['Failure mode']\n",
    "X.head()\n",
    "#y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Spliting the data into training and test sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# y.head()\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)\n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "scaler = MinMaxScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC, LinearSVC, NuSVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n",
    "\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Multi-plot confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from matplotlib import pyplot as plt\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.metrics import plot_confusion_matrix\n",
    "\n",
    "# data = load_iris()\n",
    "# X = data.data\n",
    "# y = data.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)\n",
    "classifiers = [GaussianNB(), \n",
    "               KNeighborsClassifier(n_neighbors=5),\n",
    "               LogisticRegression(solver=\"liblinear\", random_state=0),\n",
    "               SVC(probability=True, random_state=0),\n",
    "               DecisionTreeClassifier(random_state=0),\n",
    "               RandomForestClassifier(random_state=0),\n",
    "               AdaBoostClassifier(n_estimators=100, random_state=0),\n",
    "               GradientBoostingClassifier(random_state=0),\n",
    "               MLPClassifier(random_state=0)]\n",
    "\n",
    "for cls in classifiers:\n",
    "    cls.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "confusion matrix of training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,12))\n",
    "plt.rcParams['figure.dpi'] =1200\n",
    "for cls, ax in zip(classifiers, axes.flatten()):\n",
    "    plot_confusion_matrix(cls, \n",
    "                          X_train, \n",
    "                          y_train, \n",
    "                          ax=ax, \n",
    "                          cmap='Blues',\n",
    "                        display_labels=['F','FS','FS'])\n",
    "    ax.title.set_text(type(cls).__name__)\n",
    "plt.tight_layout()  \n",
    "plt.show()\n",
    "# plt.savefig('shap.png', dpi=1200)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "confusion matrix of test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,12))\n",
    "plt.rcParams['figure.dpi'] =1200\n",
    "for cls, ax in zip(classifiers, axes.flatten()):\n",
    "    plot_confusion_matrix(cls, \n",
    "                          X_test, \n",
    "                          y_test, \n",
    "                          ax=ax, \n",
    "                          cmap='Blues',\n",
    "                        display_labels=['F','FS','S'])\n",
    "    ax.title.set_text(type(cls).__name__)\n",
    "plt.tight_layout()  \n",
    "plt.show()\n",
    "# plt.savefig('test_raw.png', dpi=1200)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1 Naive Bayes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "gnb = GaussianNB()\n",
    "gnb.fit(X_train, y_train)\n",
    "print('Accuracy of GNB classifier on training set: {:.2f}'\n",
    "     .format(gnb.score(X_train, y_train)))\n",
    "print('Accuracy of GNB classifier on test set: {:.2f}'\n",
    "     .format(gnb.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "# plot_confusion_matrix(gnb, X_train, y_train, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Confusion Matrix of training set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_train = gnb.predict(X_train)\n",
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix = confusion_matrix(y_train, y_pred_train)\n",
    "print(confusion_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_train, y_pred_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Confusion Matrix of test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = gnb.predict(X_test)\n",
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix = confusion_matrix(y_test, y_pred)\n",
    "print(confusion_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2 K-Nearest neighbors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train, y_train)\n",
    "print('Accuracy of K-NN classifier on training set: {:.2f}'\n",
    "     .format(knn.score(X_train, y_train)))\n",
    "print('Accuracy of K-NN classifier on test set: {:.2f}'\n",
    "     .format(knn.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3 Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "lr = LogisticRegression(solver=\"liblinear\", random_state=0)\n",
    "lr.fit(X_train, y_train)\n",
    "print('Accuracy of lr classifier on training set: {:.2f}'\n",
    "     .format(lr.score(X_train, y_train)))\n",
    "print('Accuracy of lr classifier on test set: {:.2f}'\n",
    "     .format(lr.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4 Support Vectors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "svc=SVC(probability=True, random_state=0)\n",
    "svc.fit(X_train, y_train)\n",
    "print('Accuracy  on training set: {:.2f}'\n",
    "     .format(lr.score(X_train, y_train)))\n",
    "print('Accuracy on test set: {:.2f}'\n",
    "     .format(lr.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5 Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "dt=DecisionTreeClassifier(random_state=0)\n",
    "dt.fit(X_train, y_train)\n",
    "print('Accuracy  on training set: {:.2f}'\n",
    "     .format(lr.score(X_train, y_train)))\n",
    "print('Accuracy on test set: {:.2f}'\n",
    "     .format(lr.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6 Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rf=RandomForestClassifier(random_state=0)\n",
    "rf.fit(X_train, y_train)\n",
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(rf.score(X_train, y_train)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(rf.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7 AdaBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.datasets import make_classification\n",
    "\n",
    "ADB = AdaBoostClassifier(n_estimators=100, random_state=0)\n",
    "ADB.fit(X_train, y_train)\n",
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(ADB.score(X_train, y_train)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(ADB.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 8 Gradient Boosting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "gb=GradientBoostingClassifier(random_state=0)\n",
    "gb.fit(X_train, y_train)\n",
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(gb.score(X_train, y_train)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(gb.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 9 Neural Net"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nn = MLPClassifier(random_state=0)\n",
    "nn.fit(X_train, y_train)\n",
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(nn.score(X_train, y_train)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(nn.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the models, random forest is the one having highest accuracy. Hence, feature importances analysis is conducted to this model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model Explainablity\n",
    "\n",
    "One of the challenges of a machine leaning project is explaining the model's prediction. A model might consider some features more important than other for its prediction. Another model might weigh other features as more important. **Permutation importance** and **SHAP** are two methods one can use to understand which features were selected to have the most impact on our model's prediction.\n",
    "\n",
    "#### Permutation importance: \n",
    "The permutation importance is defined to be the **decrease in a model score** when a **single feature value** is *randomly shuffled*. The procedure breaks the relationship between the *feature* and the *target*, thus the drop in the **model score** is indicative of how much the model depends on the feature [[3](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)]. In other words, permutation importance tell us what features have the biggest impact on our model predictions. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = list(X.columns.values)\n",
    "\n",
    "importances = rf.feature_importances_\n",
    "import numpy as np\n",
    "indices = np.argsort(importances)\n",
    "\n",
    "plt.rc('font',family='Times New Roman',size=12)\n",
    "plt.title('Feature Importances')\n",
    "plt.barh(range(len(indices)), importances[indices], color='b', align='center')\n",
    "plt.yticks(range(len(indices)), [features[i] for i in indices])\n",
    "plt.xlabel('Relative Importance')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "importances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shap\n",
    "shap.initjs()\n",
    "explainer = shap.TreeExplainer(rf)\n",
    "shap_values = explainer.shap_values(X_test)\n",
    "shap.summary_plot(shap_values, X_test, feature_names=features, plot_type=\"bar\")\n",
    "\n",
    "plt.savefig('shap.png', dpi=1200)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PCA as training input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "data_pca=PCA(n_components=10).fit_transform(data_.iloc[:,:10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(data_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_pca = data_pca\n",
    "y_pca=data['Failure mode']\n",
    "# X_pca.head()\n",
    "#y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# y.head()\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.3, random_state=0)\n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "scaler = MinMaxScaler()\n",
    "X_train = scaler.fit_transform(X_train_pca)\n",
    "X_test = scaler.transform(X_test_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rf2_pca=RandomForestClassifier(random_state=0)\n",
    "rf2_pca.fit(X_train_pca, y_train_pca)\n",
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(rf2_pca.score(X_train_pca, y_train_pca)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(rf2_pca.score(X_test_pca, y_test_pca)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "plot_confusion_matrix(rf2_pca,X_train_pca, y_train_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "plot_confusion_matrix(rf2_pca,X_test_pca, y_test_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "gb=GradientBoostingClassifier(random_state=0)\n",
    "gb.fit(X_train_pca, y_train_pca)\n",
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(gb.score(X_train_pca, y_train_pca)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(gb.score(X_test_pca, y_test_pca)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "plot_confusion_matrix(gb,X_train_pca, y_train_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "plot_confusion_matrix(gb,X_test_pca, y_test_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hyperparameter Optimization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn import metrics\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## gradient boosting optimization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "params={\n",
    "   'learning_rate':np.linspace(0.05,0.25,5), \n",
    "    'max_depth':[x for x in range(1,8,1)], \n",
    "    'min_samples_leaf':[x for x in range(1,5,1)], \n",
    "    'n_estimators':[x for x in range(50,100,10)]}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = GradientBoostingClassifier()\n",
    "# clf=rf2_pca\n",
    "grid = GridSearchCV(clf, params, cv=10, scoring=\"accuracy\")\n",
    "grid.fit(X_pca, y_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid.best_score_   \n",
    "grid.best_params_  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(grid.score(X_train_pca, y_train_pca)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(grid.score(X_test_pca, y_test_pca)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "# plt.rcParams['figure.dpi'] =600\n",
    "plot_confusion_matrix(grid,X_train_pca, y_train_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "plt.rcParams['figure.dpi'] =1200\n",
    "plot_confusion_matrix(grid,X_test_pca, y_test_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## random forest optimization "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "params2={\n",
    "#    'learning_rate':np.linspace(0.05,0.25,5), \n",
    "    'criterion': ['gini','entropy'],\n",
    "    'max_features': ['auto', 'sqrt','log2'],\n",
    "    'max_depth':[x for x in range(1,8,1)], \n",
    "    'min_samples_leaf':[x for x in range(1,5,1)], \n",
    "    'n_estimators':[x for x in range(50,100,10)]}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf2 = RandomForestClassifier()\n",
    "grid2 = GridSearchCV(clf2, params2, cv=10, scoring=\"accuracy\")\n",
    "grid2.fit(X_pca, y_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid2.best_score_   \n",
    "grid2.best_params_  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Accuracy classifier on training set: {:.2f}'\n",
    "     .format(grid2.score(X_train_pca, y_train_pca)))\n",
    "print('Accuracy classifier on test set: {:.2f}'\n",
    "     .format(grid2.score(X_test_pca, y_test_pca)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "# plt.rcParams['figure.dpi'] =600\n",
    "plot_confusion_matrix(grid2,X_train_pca, y_train_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import plot_confusion_matrix\n",
    "# plt.rcParams['figure.dpi'] =1200\n",
    "plot_confusion_matrix(grid2,X_test_pca, y_test_pca, cmap='YlGnBu')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # 将降维后的数据保存为csv文件（ndarray-->DataFrame结构）\n",
    "# df_pca=pd.DataFrame(data_pca)\n",
    "# # df数据小数位数过多，为16位，将df数据全部转换为3位小数\n",
    "# print(\"*\"*100)\n",
    "# df_pca=df_pca.round(decimals=3)\n",
    "# # print(df)\n",
    "# df_pca.to_csv(\"pca_data.csv\")\n",
    "\n",
    "# # data_pca.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import numpy as np\n",
    "# from sklearn.preprocessing import scale\n",
    "# X_scaled = scale(data_pca)\n",
    "# print (X_scaled)\n",
    "# print (X_scaled.mean(axis = 0))\n",
    "# print (X_scaled.std(axis = 0))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_x=pd.DataFrame(X_scaled)\n",
    "# print(\"*\"*100)\n",
    "# df_x=df_x.round(decimals=3)\n",
    "# # print(df)\n",
    "# df_x.to_csv(\"X_scaled.csv\")\n",
    "\n",
    "# # data_pca.dtype"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "name": "Untitled11.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
