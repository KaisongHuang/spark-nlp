import html.parser


class HTMLTextParser(html.parser.HTMLParser):
    def __init__(self):
        super(HTMLTextParser, self).__init__()
        self.result = []

    def handle_data(self, d):
        self.result.append(d)

    def get_text(self):
        return ' '.join(self.result)