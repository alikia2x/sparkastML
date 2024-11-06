
import xml.etree.ElementTree as ET
    
class TTMLGenerator:
    def __init__(self, duration, xmlns="http://www.w3.org/ns/ttml", xmlns_ttm="http://www.w3.org/ns/ttml#metadata", xmlns_amll="http://www.example.com/ns/amll", xmlns_itunes="http://music.apple.com/lyric-ttml-internal"):
        self.tt = ET.Element("tt", attrib={
            "xmlns": xmlns,
            "xmlns:ttm": xmlns_ttm,
            "xmlns:amll": xmlns_amll,
            "xmlns:itunes": xmlns_itunes
        })
        self.head = ET.SubElement(self.tt, "head")
        self.metadata = ET.SubElement(self.head, "metadata")
        self.body = ET.SubElement(self.tt, "body", attrib={"dur": duration})
        self.div = ET.SubElement(self.body, "div")

    def add_lyrics(self, begin, end, agent, itunes_key, words):
        p = ET.SubElement(self.div, "p", attrib={
            "begin": begin,
            "end": end,
            "ttm:agent": agent,
            "itunes:key": itunes_key
        })
        for word, start, stop in words:
            span = ET.SubElement(p, "span", attrib={"begin": start, "end": stop})
            span.text = word

    def save(self, filename):
        tree = ET.ElementTree(self.tt)
        tree.write(filename, encoding="utf-8", xml_declaration=True)
