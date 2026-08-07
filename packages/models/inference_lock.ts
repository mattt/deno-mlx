/**
 * Per-instance FIFO lock for inference.
 * Serializes concurrent generate/embed calls on the same model
 * while leaving distinct model instances independent.
 */

export class InferenceLock {
  #tail: Promise<void> = Promise.resolve();

  /**
   * Acquire exclusive access.
   * Resolves to a `release` function.
   * If `signal` aborts while queued, rejects with AbortError without taking the lock.
   */
  acquire(signal?: AbortSignal): Promise<() => void> {
    signal?.throwIfAborted();

    let release!: () => void;
    const prev = this.#tail;
    this.#tail = new Promise<void>((r) => {
      release = r;
    });

    return prev.then(() => {
      signal?.throwIfAborted();
      return release;
    });
  }

  /** Run `fn` exclusively under this lock. */
  async run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const release = await this.acquire(signal);
    try {
      return await fn();
    } finally {
      release();
    }
  }
}
