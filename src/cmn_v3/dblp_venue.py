#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DblpVenue:
    """
    Class representing a venue in academic publications
    """

    def __init__(self, id, raw):
        """
        Initialize a DblpVenue object

        Args:
            id: Unique identifier for the venue
            raw: Raw venue string from DBLP
        """

        self.id = id
        self.raw = raw

    def __str__(self):
        """
        String representation of the venue
        
        Returns:
            String with venue information
        """
        return self.raw
    
    def __repr__(self):
        """
        Formal representation of the venue
        
        Returns:
            String representation for debugging
        """
        return f"Venue(id={self.id}, raw={self.raw})"