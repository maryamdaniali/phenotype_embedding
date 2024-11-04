import math

class Mist:
    """
    A Data Structure to store a minimum set of disjoint intervals (Mist).
    It supports adding intervals, merging them, and checking for intersections.
    """
    def __init__(self):
        """
        Initializes the Mist structure with default guard intervals (-∞, ∞).
        These guard intervals prevent boundary issues when adding or merging intervals.
        """
        self.data = [[-math.inf,-math.inf],[math.inf, math.inf]]
    def __len__(self):  
        """
        Returns (int) number of intervals in the structure.
        """
        return len(self.data)

    def __str__(self):    
        """
        Returns a string (str) representation of the interval data, used for printing.
        """
        return str(self.data)

    def guards(self,x):                                             
        """
        Checks the validity of the input interval.

        Args:
            x (list): Interval to be checked, a list of two integers [start, end].

        Raises:
            Exception: If the input is not a list, has an invalid size, 
                       or if the interval is not valid (start > end).
        """
        if not isinstance(x, list):
            raise Exception('Not a List')
        if len(x) != 2:
            raise Exception('Interval is not the correct size')
        if not isinstance(x[0], int):
            raise Exception('d is not an Int')
        if not isinstance(x[1], int):
            raise Exception('f is not an Int')
        if x[0] > x[1]:
            raise Exception('D > F')

    def unravel(self):                                  
        """
        Flattens the intervals into a list of individual values.
        Example: [1,2],[6,7] is 1, [1,2, ..., 7]
        Returns:
            list: A list of all individual integers covered by the intervals.
        """
        out = []
        for i in range(1, len(self.data)-1):
            for j in range(self.data[i][0],self.data[i][1]+1):
                out.append(j)
        return out

    def merge(self, mist_outside):        
        """
        Merges another Mist object into the current one.

        Args:
            mist_outside (Mist): Another Mist structure to be merged into the current one.
        """
        for x in mist_outside.data[1:len(mist_outside)-1]:
            self.insert(x)

    def intersects(self, x):
        """
        Checks if an integer or list of integers intersects with any interval in the structure.

        Args:
            x (int or list): A single integer or a list of integers.

        Returns:
            bool: True if any of the integers intersect with an interval, else False.
        """             

        if isinstance(x, int):
            return self.intersects1(x)
        if isinstance(x, list):
            for n in x:
                if self.intersects1(n):
                    return True
        return False        

    def intersects1(self, x, i=0, pos = None, m = None):          
        """
        Binary search to check if an integer intersects with any interval.

        Args:
            x (int): The integer to check for intersection.
            i (int): The recursion depth (for controlling binary search).
            pos (int): The current position in the interval list.
            m (int): The total number of intervals.

        Returns:
            bool: True if the integer intersects, else False.
        """    
        i += 1
        if pos is None:
            m = len(self)
            pos  = max( math.ceil( m / (2 **  i  ) )  - 1,1)
        step = max( math.ceil( m / (2 ** (i+1) )) - 1,1)
        if x < self.data[pos][0]:
            return self.intersects1(x,i,pos-step,m)
        elif x >= self.data[pos+1][0]:           
            return self.intersects1(x,i,pos+step,m) 
        elif x > self.data[pos][1]:
            return False
        else:
            return True            
    
    def insert(self,x):                                        
        """
        Inserts a new interval into the Mist structure, merging it with overlapping intervals.

        Args:
            x (list): A list representing the new interval [start, end].
        """
        self.guards(x)                                              
        d, f = x
        pos = max( math.ceil(len(self.data)/ (1 * 2)) - 1,1)
        d_pos = self.fnd_kpos(d,1,pos)                             
        f_pos = self.fnd_kpos(f,1,pos,'f') 
        # Merge the new interval with overlapping intervals                           
        d_min = min(d, self.data[math.ceil (d_pos)][0])             
        f_max = max(f, self.data[math.floor(f_pos)][1]) 
        # Remove intervals covered by the new one           
        del self.data[math.ceil(d_pos):math.ceil(f_pos+.5)]        
        self.data.insert(math.ceil(d_pos),[d_min,f_max])           
        return

    def fnd_kpos(self, d, i, pos, type='d'):
        """
        Finds the correct position of a given integer in the interval structure using binary search.

        Args:
            d (int): The integer to find.
            i (int): The recursion depth (for binary search).
            pos (int): The current position in the interval list.
            type (str): Either 'd' for start or 'f' for end, indicating which boundary to find.

        Returns:
            float or int: The position of the integer in the interval structure.
        """
        i += 1
        step = max( math.ceil(len(self.data)/ (2 ** i)) - 1,1)
        if self.data[pos][0] <= d: 
            if type == 'f':
                if d == self.data[pos+1][0] - 1:
                    return pos + 1
            if d <= self.data[pos][1] + 1:
                return pos
            if d <= self.data[pos+1][0] - 1: 
                    return pos + .5
            return self.fnd_kpos(d,i,pos+step, type)
        return self.fnd_kpos(d,i,pos-step, type)