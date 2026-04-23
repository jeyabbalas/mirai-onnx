export const MISSING_VALUE = -1;

export function coerceInt(value: number | boolean | null | undefined): number {
  if (value === undefined || value === null) return MISSING_VALUE;
  if (value === true) return 1;
  if (value === false) return 0;
  return value | 0;
}
