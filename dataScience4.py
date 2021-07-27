from typing import final
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
import seaborn as sns
import pandas as pd
import csv

rows = []

with open('Stars.csv', 'r') as f:
    csvreader = csv.reader(f)

    for i in csvreader:
        rows.append(i)

headers = rows[0]

star_data = rows[1:]

final_list = []

df = pd.read_csv("stars.csv")

solar_mass_list = df["solar_mass"].tolist()

solar_radius_list = df["solar_radius"].tolist()

solar_mass_list.pop(0)

solar_radius_list.pop(0)

star_solar_mass_si_unit = []

for data in solar_mass_list:
    si_unit = float(data) * 1.989e+30
    star_solar_mass_si_unit.append(si_unit)

print(star_solar_mass_si_unit)

star_solar_radius_si_unit = []

for data in solar_radius_list:
    si_unit = float(data) * 6.957e+8
    star_solar_radius_si_unit.append(si_unit)

print(star_solar_radius_si_unit)

star_masses = star_solar_mass_si_unit

star_radiuses = star_solar_radius_si_unit

star_names = df["star_names"].tolist()

star_names.pop(0)

star_gravities = []

for index, data in enumerate(star_names):
    gravity = (float(star_masses[index]) * 5.972e+24) / (float(star_radiuses[index]) * float(star_radiuses[index]) * 6371000 * 6371000) * 6.674e-11
    star_gravities.append(gravity)

print(star_gravities)

star_masses = []

star_radiuses = []

star_gravities = []

for star in star_data:
    if star[3] == '?' or star[4] == '?' or star[5] == '?':
        star_data.remove(star)
    else:
        star_masses.append(float(star[3]))
        star_radiuses.append(float(star[4]))
        star_gravities.append(float(star[5]))

fig = px.scatter(x = star_masses, y = star_radiuses, size = star_gravities, range_y = (-2e+8, 3e+9), range_x = (-1e+31, 2.1e+32), labels = dict(x = 'Mass of Star', y = 'Radius of Star'))

fig.show()

for star in star_data:
    star_masses.append(star[3])
    star_radiuses.append(star[4])

X = []

for index, star_mass in enumerate(star_masses):
  temp_list = [star_radiuses[index], star_mass]
  X.append(temp_list)

WCSS = []

for i in range(1, 11):
  k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  k_means.fit(X)
  WCSS.append(k_means.inertia_)

plt.figure(figsize = (10, 5))

sns.lineplot(range(1, 11), WCSS, marker = 'o', color = 'red')

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.show()

for star in star_data:
    if float(star[2]) < 100:
        if float(star[5]) > 150 and float(star[5]) < 350:
            final_list.append(star)

Star_names = []

Distance = []

Mass = []

Radius = []

Gravity = []

for i in final_list:
    Star_names.append(i[1])
    Distance.append(i[2])
    Mass.append(i[3])
    Radius.append(i[4])
    Gravity.append(i[5])

df = pd.DataFrame(
    list(zip(Star_names, Distance, Mass, Radius, Gravity)),
    columns=["Star Name", "Distance", "Mass", "Radius", "Gravity"],
)

df.to_csv('Final.csv')