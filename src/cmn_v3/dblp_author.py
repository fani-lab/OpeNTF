#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DblpAuthor:
    """
    Class representing an author in academic publications
    
    This class is specifically used for DBLP dataset processing.
    """
    
    def __init__(self, id, name, org=""):
        """
        Initialize an Author object
        
        Args:
            id: Unique identifier for the author
            name: Name of the author
            org: Organization/affiliation of the author (optional)
        """
        self.id = id
        self.name = name
        self.org = org
    
    def __str__(self):
        """
        String representation of the author
        
        Returns:
            String with author information
        """
        if self.org:
            return f"{self.name} ({self.id}) - {self.org}"
        return f"{self.name} ({self.id})"
    
    def __repr__(self):
        """
        Formal representation of the author
        
        Returns:
            String representation for debugging
        """
        return f"DblpAuthor(id={self.id}, name={self.name}, org={self.org})" 