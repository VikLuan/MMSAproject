from urllib import request  # requires python3

KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"


def load_kinetics_labels():
    with request.urlopen(KINETICS_URL) as obj:
        labels = [line.decode("utf-8").strip() for line in obj.readlines()]
    print("Found in total %d labels." % len(labels))
    return labels


def set_categories(ucf_videos):
    categories = {}
    for video in ucf_videos:
        category = video[2:-12]
        if category not in categories:
            categories[category] = []
        categories[category].append(video)
    print("Found in total %d videos in overall %d categories." % (len(ucf_videos), len(categories)))
'''
print("\n")
head1 = "CATEGORY"
head2 = "No. of Videos"
head3 = "Details"
print(" ", head1, " \t  ", head2, " \t\t\t ", head3)
for category, sequences in categories.items():
    summary = ", ".join(sequences[:2])
    print("%-20s    %4d           %s, ..." % (category, len(sequences), summary))


def getLabels():
    return labels
'''