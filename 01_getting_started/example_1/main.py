import stan

with open("code.stan", "r") as f:
  schools_code = f.read()


schools_data = {"J": 8,
                "y": [28,  8, -3,  7, -1,  1, 18, 12],
                "sigma": [15, 10, 16, 11,  9, 11, 10, 18]}

posterior = stan.build(schools_code, data=schools_data, random_seed=1)

fit = posterior.sample(num_chains=4, num_samples=1000)

eta = fit["eta"]  # array with shape (8, 4000)

print("Done")
print("eta ")
print(eta)