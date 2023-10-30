import os
import warnings
warnings.filterwarnings("ignore")

os.chdir('scripts')
os.system('python Save_features_and_user_scores_W4.py')
print('features and scores W4 done')
os.system('python creation_of_features_for_synthetic_users.py')
print('features for synthetic users done')
os.system('python fitting_of_models_to_real_users.py')
print('fitting models to real users done')
os.system('python Bound_with_logistic.py')
print('bound with logistic done')
os.system('python Synthetic_users_give_score_to_generated_experiences.py')
print('synthetic users give score to generated experiences done')










