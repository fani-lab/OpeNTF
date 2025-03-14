#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class GithContributor:
    """
    Class representing a Gith contributor
    
    This class encapsulates contributor data from Gith repositories.
    """
    
    def __init__(self, id, login, contributions=0):
        """
        Initialize a Gith contributor
        
        Args:
            id: Unique identifier for the contributor
            login: Username/login of the contributor
            contributions: Number of contributions (optional)
        """
        self.id = id
        self.login = login
        self.contributions = contributions
    
    def __str__(self):
        """
        String representation of the contributor
        
        Returns:
            String with contributor information
        """
        return f"{self.login} ({self.id}) - {self.contributions} contributions"
    
    def __repr__(self):
        """
        Formal representation of the contributor
        
        Returns:
            String representation for debugging
        """
        return f"GithContributor(id={self.id}, login={self.login}, contributions={self.contributions})" 