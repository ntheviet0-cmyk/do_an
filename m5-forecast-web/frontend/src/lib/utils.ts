import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
export function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)) }
export const fmt = (n: number | null | undefined, digits = 2) => n == null ? '—' : new Intl.NumberFormat('vi-VN', { maximumFractionDigits: digits }).format(n)
