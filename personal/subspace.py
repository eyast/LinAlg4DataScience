import numpy as np
import time
import plotly.graph_objects as go

SIZE = 100

vec1 = np.array([3, 5, 1])
vec2 = np.array([0, 2, 2])
vecs = np.vstack([vec1, vec2]).T

random_scalars = np.random.uniform(-4, 4, (SIZE, 2))
time.sleep(1)

results = np.zeros((100, 3))

results = random_scalars @ vecs.T

fig = go.Figure(data=go.Scatter3d(
    x=results[:,0],
    y=results[:,1],
    z=results[:,2],
    mode='markers'
))
fig.show()