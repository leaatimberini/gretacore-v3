#!/usr/bin/env python3
import sys

try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    print(f"STATUS=SKIPPED reason=missing_deps err={e}")
    sys.exit(0)

# Basic backend sanity check.
try:
    devices = jax.devices()
    _ = devices[0]
except Exception as e:
    print(f"STATUS=SKIPPED reason=no_backend err={e}")
    sys.exit(0)

@jax.jit
def add_one(x):
    return x + 1

x = jnp.arange(8, dtype=jnp.float32)
y = add_one(x)
if jnp.all(y == x + 1):
    print("STATUS=OK")
else:
    print("STATUS=FAILED")
    sys.exit(1)

# Custom call hook placeholder (LOE-5).
print("CUSTOM_CALL_STATUS=TODO")
