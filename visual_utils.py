# libraries
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


# functions
def visualize_horizontal_bar_chart(places, distances, curr_query, curr_time, curr_user):
    plt.rcdefaults()
    title = 'query : '+str(curr_query)+' time : '+str(curr_time)+' user : '+str(curr_user)
    places_input = places
    y_pos = np.arange(len(places_input))
    distances_input = distances
    plt.barh(y_pos, distances_input, align='center', alpha=0.5)
    plt.yticks(y_pos, places_input)
    plt.xlabel('distance (km)')
    plt.ylabel('place name')
    # plt.title('Distance of recommendation places')
    plt.title(title)
    plt.show()


def visualize_histogram_all_distances(all_distance):
    plt.figure(figsize=(10, 10))
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 60, 80, 100]
    hist, bins = np.histogram(all_distance[0:47800], bins=bins)
    width = 0.7 * (bins[5] - bins[4])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Distance (km)')
    plt.ylabel('Number of POI')
    plt.show()


def histogram(all_z_scores):
    plt.figure(figsize=(7, 7))
    bins = [-3.5, -3.4, -3.2, -3.1, -3.0, -2.5, -2, -1.5, -1, -0.5, 0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0,
            1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    hist, bins = np.histogram(all_z_scores[0:47800], bins=bins)
    width = 0.6 * (bins[5] - bins[4])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Z-score')
    plt.ylabel('Number of POI')
    plt.show()


def test_word_cloud(input_dictionary):
    from collections import Counter
    word_could_dict = Counter(input_dictionary)
    my_wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(word_could_dict)

    plt.figure(figsize=(15, 8))
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.show()
    # plt.savefig('yourfile.png', bbox_inches='tight')
    plt.close()


def show_POIs_on_map():
    print("show list of POIs in map")