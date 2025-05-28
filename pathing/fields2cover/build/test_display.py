import matplotlib
matplotlib.use('QtAgg')  # Try QtAgg instead of Qt5Agg
import matplotlib.pyplot as plt
import numpy as np

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Test Plot - You should see this window')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
