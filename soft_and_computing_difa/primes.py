
def find_primes(n):
    
    prime_list = []
    prime = [True for i in range(n+1)] 
    p = 2
    
    while (p**2 <= n):
        if (prime[p] == True): 
            for i in range(p * 2, n+1, p): 
                prime[i] = False
        p += 1
    
    
    for p in range(2, n): 
        if prime[p]: 
            prime_list.append(p)
            print(p)
            
            
    return prime_list
 #%%           

find_primes(10)

