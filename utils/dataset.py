from csv import DictReader


class DataSet():
     def __init__(self, name="train", path="fnc-1"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        unrelated_stance = []
        related_stance = []

        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])
            if s['Stance'] == 'unrelated':
                unrelated_stance.append(s)

            else:
                related_stance.append(s)

        self.related_stance = related_stance
        self.unrelated_stance = unrelated_stance

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))
        print("Total unrelated stances: " + str(len(self.unrelated_stance)))
        print("Total related stances: " + str(len(self.related_stance)))



    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
