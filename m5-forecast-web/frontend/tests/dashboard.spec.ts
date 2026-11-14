import { test, expect } from '@playwright/test'

test('real forecast → chart → metrics → CSV export', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message))
  await page.goto('/forecast')
  await expect(page.getByRole('button', { name: 'Chạy dự báo' })).toBeEnabled()
  await expect(page.getByLabel('Cửa hàng', { exact: true })).toHaveValue('TX_1')
  await page.getByRole('button', { name: 'Chạy dự báo' }).click()
  await expect(page.getByRole('heading', { name: 'TFT · FOODS_3_586_TX_1' })).toBeVisible()
  await expect(page.getByTestId('chart')).toBeVisible()
  await expect(page.getByText('WMAPE ↓', { exact: true })).toBeVisible()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Xuất CSV' }).click()
  expect((await download).suggestedFilename()).toBe('m5-forecast.csv')
  expect(errors).toEqual([])
})

test('dependent selectors and model selection', async ({ page }) => {
  await page.goto('/forecast')
  await expect(page.getByRole('button', { name: 'Chạy dự báo' })).toBeEnabled()
  await page.getByLabel('Bang', { exact: true }).selectOption('WI')
  await expect(page.getByLabel('Cửa hàng', { exact: true })).toHaveValue('WI_1')
  const storeOptions=await page.getByLabel('Cửa hàng', { exact: true }).locator('option').allTextContents()
  expect(storeOptions.every(s=>s.startsWith('WI_'))).toBeTruthy()
  await page.getByLabel('Mô hình', { exact: true }).selectOption('LSTM')
  await expect(page.getByLabel('Calibration', { exact: true }).locator('option')).toHaveCount(1)
  await page.getByRole('button', { name: 'Chạy dự báo' }).click()
  await expect(page.getByRole('heading', { name: /^LSTM ·/ })).toBeVisible()
})

test('all routes and mobile navigation', async ({ page }) => {
  for (const [route,title] of [['/','Tổng quan'],['/comparison','So sánh mô hình'],['/hierarchy','Phân tích phân cấp'],['/calibration','Hiệu chỉnh dự báo'],['/explainability','Giải thích mô hình'],['/system','Thông tin hệ thống']]) {
    await page.goto(route)
    await expect(page.getByRole('heading', { name:title, exact:true })).toBeVisible()
    await expect(page.locator('main')).not.toContainText('Không thể hiển thị trang')
  }
  await page.setViewportSize({width:390,height:844})
  await page.getByRole('button',{name:'Mở menu'}).click()
  await page.getByRole('link',{name:'Dự báo nhu cầu Demand Forecast'}).click()
  await expect(page.getByRole('heading',{name:'Dự báo nhu cầu',exact:true})).toBeVisible()
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy()
})

test('comparison and coherent hierarchy drill-down', async ({page}) => {
  await page.goto('/comparison')
  await expect(page.getByRole('button',{name:'Chạy dự báo'})).toBeEnabled()
  await page.getByRole('button',{name:'Chạy dự báo'}).click()
  await expect(page.getByRole('heading',{name:'Sai số trên mục tiêu đã chọn'})).toBeVisible()
  await expect(page.locator('.best-row')).toHaveCount(1)
  await page.goto('/hierarchy')
  await expect(page.getByRole('button',{name:'Chạy dự báo'})).toBeEnabled()
  await page.getByRole('button',{name:'Chạy dự báo'}).click()
  await expect(page.getByText('Max daily difference',{exact:true})).toBeVisible()
  await page.getByRole('button',{name:'Drill TX',exact:true}).click()
  await expect(page.getByRole('heading',{name:'L2 · TX',exact:true})).toBeVisible()
})

test('API error is visible and loading state recovers', async ({page}) => {
  await page.goto('/forecast')
  await expect(page.getByRole('button',{name:'Chạy dự báo'})).toBeEnabled()
  await page.route('**/api/forecast', async route => {
    await new Promise(resolve=>setTimeout(resolve,300))
    await route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Artifact unavailable for verification'})})
  })
  await page.getByRole('button',{name:'Chạy dự báo'}).click()
  await expect(page.getByRole('status')).toBeVisible()
  await expect(page.getByRole('alert')).toContainText('Artifact unavailable for verification')
  await expect(page.getByRole('button',{name:'Chạy dự báo'})).toBeEnabled()
})
