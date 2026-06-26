"""Carbon-credit auction WDP — classical clearing engine (demo build, no ORBIT)."""
from .generator import (AuctionInstance, generate_instance, generate_real_instance,
                        instance_stats)
from .qubo import QUBO, build_qubo, decode
from .solvers import exact_mip, greedy, simulated_annealing

__all__ = ["AuctionInstance", "generate_instance", "generate_real_instance", "instance_stats",
           "QUBO", "build_qubo", "decode", "exact_mip", "greedy", "simulated_annealing"]
