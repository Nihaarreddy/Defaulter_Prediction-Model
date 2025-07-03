import data_ingest
import preprocessing
from sklearn.model_selection import train_test_split
import model_selection
import train_save_model
df=data_ingest.data_load('Bank_Personal_Loan_Modelling.xlsx','Data')

new_df=data_ingest.data_load('Bank_Personal_Loan_Modelling.xlsx','Data')


df= preprocessing.preprocess(df)

X = df.drop(['Personal Loan', 'ID','ZIP Code'], axis=1)
y = df['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


 
# Run model evaluation
model_selection.evaluate_models_with_grid_search(X_train, X_test, y_train, y_test,new_df)
#  Find best model

best_model, best_params = model_selection.evaluate_models_with_grid_search(X_train, X_test, y_train, y_test)
 #Train on full data and save

train_save_model.train_and_save_final_model(X, y, best_model, model_name="best_final_model.pkl")
 