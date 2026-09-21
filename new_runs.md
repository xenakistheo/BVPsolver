For all ablations do it so I can do a job array with different seeds. 

### Ablation nr. 1. 
For all problems, run with seeds 20-29. 
PROBLEMS=(pollu rober vanderpol hires orego davis-skodje)
MODELS=(stiff mlp GELU-scaled)

for pretraining and training, I want to test derivative matching with shooting vs derivative matching with collocaiton. 
In total therefore, we have 2 diffferent training regimes, 3 models, and 10 seeds. This means 2*3*10 = 60 runs per problem, and 60*6 = 360 runs in total.




### Ablation nr. 2
Run with seeds 30-39. 
Only run rober. 
I want to train all three different models, mpl, GELU-scaled and stiff. For pretraining and training, I want to test all 
different combinations with pretraining = [derivative matching, none]
training = [none, shooting, collocation] (except for none, none). In total there are therefore 5 training regimes, 3 models, and 10 seeds. This means 5*3*10 = 150 runs in total.


### Ablation nr. 3
Run with seeds 40-49.
Only run davis skodje. Let epsilon vary between 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6. 
I only want to compare stiff model with derivative matching and collocation, vs. gelu with derivative matching and shooting. In total there are therefore 2 model/training combiations, 6 epsilons, and 10 seeds. This means 2*6*10 = 120 runs in total.

### Ablation nr. 4
Run with seeds 40-49
Only run van der pol. Let mu vary between 1, 5, 10, 50, 100, 150, 200
I only want to compare stiff model with derivative matching and collocation, vs. gelu with derivative matching and shooting. In total there are therefore 2 model/training combiations, 7 mu's, and 10 seeds. This means 2*7*10 = 140 runs in total.