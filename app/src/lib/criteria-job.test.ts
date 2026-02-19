import { describe, expect, it, vi } from "vitest"
import { resolveCriteriaJob } from "@/lib/criteria-job"

describe("resolveCriteriaJob", () => {
  it("returns completed scopes after polling", async () => {
    const createJob = vi.fn().mockResolvedValue({ jobId: "job-1" })
    const getStatus = vi
      .fn()
      .mockResolvedValueOnce({ status: "pending" })
      .mockResolvedValueOnce({ status: "completed" })
    const getResult = vi.fn().mockResolvedValue({
      data: { scope: { criterion: { weight: 100, description: "desc" } } },
    })
    const normalize = vi.fn().mockReturnValue([{ id: "scope-1" }])
    const sleep = vi.fn().mockResolvedValue(undefined)

    const result = await resolveCriteriaJob({
      createJob,
      getStatus,
      getResult,
      normalize,
      sleep,
      pollIntervalMs: 1,
    })

    expect(result).toEqual({
      status: "completed",
      scopes: [{ id: "scope-1" }],
    })
    expect(createJob).toHaveBeenCalledTimes(1)
    expect(getStatus).toHaveBeenCalledTimes(2)
    expect(getResult).toHaveBeenCalledTimes(1)
    expect(normalize).toHaveBeenCalledTimes(1)
    expect(sleep).toHaveBeenCalledTimes(1)
  })

  it("returns failed when job fails", async () => {
    const createJob = vi.fn().mockResolvedValue({ jobId: "job-2" })
    const getStatus = vi.fn().mockResolvedValue({ status: "failed" })
    const getResult = vi.fn()
    const normalize = vi.fn()

    const result = await resolveCriteriaJob({
      createJob,
      getStatus,
      getResult,
      normalize,
    })

    expect(result).toEqual({ status: "failed" })
    expect(getResult).not.toHaveBeenCalled()
    expect(normalize).not.toHaveBeenCalled()
  })

  it("returns empty when result normalization yields no scopes", async () => {
    const createJob = vi.fn().mockResolvedValue({ jobId: "job-3" })
    const getStatus = vi.fn().mockResolvedValue({ status: "completed" })
    const getResult = vi.fn().mockResolvedValue({ data: { invalid: true } })
    const normalize = vi.fn().mockReturnValue([])

    const result = await resolveCriteriaJob({
      createJob,
      getStatus,
      getResult,
      normalize,
    })

    expect(result).toEqual({ status: "empty" })
    expect(getResult).toHaveBeenCalledTimes(1)
    expect(normalize).toHaveBeenCalledTimes(1)
  })

  it("returns cancelled when job is cancelled", async () => {
    const createJob = vi.fn().mockResolvedValue({ jobId: "job-4" })
    const getStatus = vi.fn().mockResolvedValue({ status: "cancelled" })
    const getResult = vi.fn()
    const normalize = vi.fn()

    const result = await resolveCriteriaJob({
      createJob,
      getStatus,
      getResult,
      normalize,
    })

    expect(result).toEqual({ status: "cancelled" })
    expect(getResult).not.toHaveBeenCalled()
    expect(normalize).not.toHaveBeenCalled()
  })
})
