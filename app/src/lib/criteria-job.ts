export type CriteriaJobStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed"
  | "cancelled"

type JobCreateResult = { jobId: string }
type JobStatusResult = { status: CriteriaJobStatus }
type JobResult = { data: unknown }

type ResolveCriteriaJobOptions<TScope> = {
  createJob: () => Promise<JobCreateResult>
  getStatus: (jobId: string) => Promise<JobStatusResult>
  getResult: (jobId: string) => Promise<JobResult>
  normalize: (raw: unknown) => TScope[]
  pollIntervalMs?: number
  sleep?: (ms: number) => Promise<void>
}

export type ResolveCriteriaJobResult<TScope> =
  | { status: "completed"; scopes: TScope[] }
  | { status: "failed" }
  | { status: "cancelled" }
  | { status: "empty" }

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export async function resolveCriteriaJob<TScope>({
  createJob,
  getStatus,
  getResult,
  normalize,
  pollIntervalMs = 2000,
  sleep = defaultSleep,
}: ResolveCriteriaJobOptions<TScope>): Promise<ResolveCriteriaJobResult<TScope>> {
  const { jobId } = await createJob()

  while (true) {
    const { status } = await getStatus(jobId)

    if (status === "completed") {
      const result = await getResult(jobId)
      const scopes = normalize(result.data)
      if (scopes.length === 0) {
        return { status: "empty" }
      }
      return { status: "completed", scopes }
    }

    if (status === "failed") {
      return { status: "failed" }
    }

    if (status === "cancelled") {
      return { status: "cancelled" }
    }

    await sleep(pollIntervalMs)
  }
}
