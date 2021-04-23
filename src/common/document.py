from common.team import Team
class Document(Team):
    def __init__(self, id, authors, title, doc_type, year, venue, references, fos, keywords):
        super().__init__(id, authors)
        self.title = title
        self.year = year
        self.doc_type = doc_type
        self.venue = venue
        self.references = references
        self.fos = fos
        self.keywords = keywords
        self.fields = self.set_fields()
        
    # Fill the fields attribute with non-zero weight from FOS
    def set_fields(self):
        fields = []
        for field in self.fos:
            if field["w"] != 0.0:
                fields.append(field["name"])
        # Extend the fields with keywords
        if len(self.keywords) != 0:
            fields.extend(self.keywords)
        return fields
    
    def get_fields(self):
        return self.fields
    