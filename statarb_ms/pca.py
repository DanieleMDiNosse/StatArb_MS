from factors import *
from mpl_toolkits.axisartist.axislines import SubplotZero
from mpl_toolkits import mplot3d
plt.style.use('seaborn')

fig = plt.figure()
# ax0 = SubplotZero(fig, 211)
ax1 = SubplotZero(fig, 111)
# fig.add_subplot(ax0)
fig.add_subplot(ax1)

for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
    ax1.axis[direction].set_axisline_style("-|>")
    # ax0.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax1.axis[direction].set_visible(True)
    # ax0.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax1.axis[direction].set_visible(False)
    # ax0.axis[direction].set_visible(False)

x1 = np.random.normal(0, 4, size=100)
x2 = 1.5*x1 + np.random.normal(0,6, size=100)
x3 = x1+x2 + np.random.normal(0,6, size=100)

scaler = StandardScaler()
x1 = scaler.fit_transform(x1.reshape(-1,1))
x2 = scaler.fit_transform(x2.reshape(-1,1))
x3 = scaler.fit_transform(x3.reshape(-1,1))
# ax0.scatter(x1,x2, s=15, c='blue', alpha=0.7)
# ax0.title.set_text('Original space \n')
# ax0.set_xlim([-3.5, 3.5])
# ax0.set_ylim([-3, 3])

df = pd.DataFrame()
df['x1'] = [x[0] for x in x1]
df['x2'] = [x[0] for x in x2]
df['x3'] = [x[0] for x in x3]

pca = PCA(n_components=2)
pca.fit(df)
comp = pca.fit_transform(df)
print(pca.explained_variance_ratio_.sum())

ax1.scatter(comp[:,0], comp[:,1], s=15, c='blue', alpha=0.7)
ax1.title.set_text('Transformed space \n')
ax1.set_xlim([-3.5, 3.5])
ax1.set_ylim([-3, 3])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['x1'],df['x2'],df['x3'], color='blue', alpha=0.7)
plt.title('Original space')
plt.show()
