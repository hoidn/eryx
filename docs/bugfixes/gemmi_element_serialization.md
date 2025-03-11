# Gemmi Element Serialization Bug Fix

## Issue Description

When serializing and deserializing Gemmi Element objects, the system was incorrectly handling the element symbol. The string representation of a Gemmi Element object is in the format `<gemmi.Element: X>` where X is the actual element symbol (e.g., C, N, O). This representation was being stored directly in the serialized data, causing two problems:

1. The serialized element symbol contained the full string representation (`<gemmi.Element: N>`) instead of just the element symbol (`N`)
2. During deserialization, this incorrect format was passed to the Gemmi Element constructor, causing incorrect element assignment

## Impact

This bug affected:
- State-based testing that relies on serialized Gemmi objects
- Proper element identification in structure analysis
- Weight and atomic number properties of elements

## Root Cause Analysis

The issue was in the `serialize_element` and `deserialize_element` methods in `eryx/autotest/gemmi_serializer.py`:

1. In `serialize_element`, the method was directly using `str(element)` which returns `<gemmi.Element: X>` instead of extracting just the element symbol
2. In `deserialize_element`, the method wasn't checking if the symbol was in the full string format before passing it to the Gemmi Element constructor

## Solution

The fix involved:

1. Using regex to extract just the element symbol from the string representation
2. Adding proper error handling and fallbacks
3. Ensuring consistent handling in both serialization and deserialization

### Key Changes

In `serialize_element`:
- Added regex pattern to extract just the element symbol from `str(element)`
- Stored the clean symbol in the serialized data

In `deserialize_element`:
- Added regex pattern to handle cases where the full string representation was stored
- Added better error handling and fallbacks

## Testing

The fix was verified by:
- Running the `test_serialize_element` and `test_serialize_deserialize` tests in `tests/test_gemmi_serializer.py`
- Confirming that element symbols are correctly extracted and restored
- Verifying that element properties like weight and atomic number are preserved

## Lessons Learned

1. When serializing objects with custom string representations, always extract the actual data needed rather than using the string representation directly
2. Add proper validation and cleaning of data during both serialization and deserialization
3. Include comprehensive tests that verify both the format of serialized data and the correctness of deserialized objects
