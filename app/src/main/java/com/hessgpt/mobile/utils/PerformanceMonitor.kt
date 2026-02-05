package com.hessgpt.mobile.utils

import kotlin.math.roundToInt

/**
 * Moniteur de performances pour l'inférence
 */
class PerformanceMonitor {
    
    private var startTime: Long = 0
    private var endTime: Long = 0
    private var tokenCount: Int = 0
    private var firstTokenTime: Long = 0
    
    /**
     * Démarre le monitoring d'une génération
     */
    fun startGeneration() {
        startTime = System.currentTimeMillis()
        endTime = 0
        tokenCount = 0
        firstTokenTime = 0
    }
    
    /**
     * Enregistre un token généré
     */
    fun recordToken() {
        tokenCount++
        if (firstTokenTime == 0L) {
            firstTokenTime = System.currentTimeMillis()
        }
    }
    
    /**
     * Termine le monitoring
     */
    fun endGeneration() {
        endTime = System.currentTimeMillis()
    }
    
    /**
     * Obtient les statistiques de performance
     */
    fun getStats(): PerformanceStats {
        val totalTime = if (endTime > 0) endTime - startTime else 0
        val generationTime = if (endTime > 0 && firstTokenTime > 0) {
            endTime - firstTokenTime
        } else 0
        
        val timeToFirstToken = if (firstTokenTime > 0) {
            firstTokenTime - startTime
        } else 0
        
        val tokensPerSecond = if (generationTime > 0 && tokenCount > 0) {
            (tokenCount.toFloat() / generationTime * 1000).roundToInt() / 10f
        } else 0f
        
        return PerformanceStats(
            totalTokens = tokenCount,
            totalTimeMs = totalTime,
            timeToFirstTokenMs = timeToFirstToken,
            tokensPerSecond = tokensPerSecond,
            latencyMs = totalTime
        )
    }
    
    /**
     * Réinitialise le moniteur
     */
    fun reset() {
        startTime = 0
        endTime = 0
        tokenCount = 0
        firstTokenTime = 0
    }
}

/**
 * Statistiques de performance
 */
data class PerformanceStats(
    val totalTokens: Int,
    val totalTimeMs: Long,
    val timeToFirstTokenMs: Long,
    val tokensPerSecond: Float,
    val latencyMs: Long
) {
    override fun toString(): String {
        return "Tokens: $totalTokens | " +
               "Vitesse: $tokensPerSecond tok/s | " +
               "1er token: ${timeToFirstTokenMs}ms | " +
               "Total: ${totalTimeMs}ms"
    }
}
