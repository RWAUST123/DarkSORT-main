import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

mean = [0, 0]
cov = [[1, 0.7], [0.7, 1]]
x, y = np.random.multivariate_normal(mean, cov, 500).T
inverse_cov = np.linalg.inv(cov)
mahalanobis_distance = []
for point in zip(x, y):
    mahalanobis_distance.append(np.sqrt(np.dot(np.dot(point - mean, inverse_cov), point - mean)))
plt.scatter(x, y, c=mahalanobis_distance, cmap='viridis', label='')
confidence_interval = chi2.ppf(0.95, df=2)  # 95%置信区间
mahalanobis_contours = plt.contour(x, y, mahalanobis_distance, levels=[confidence_interval], colors='r', linestyles='dashed', label='Mahalanobis距离')
