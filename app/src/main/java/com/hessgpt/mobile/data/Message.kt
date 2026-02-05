package com.hessgpt.mobile.data

/**
 * Classe de données représentant un message du chat
 */
data class Message(
    var text: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis(),
    val isSystem: Boolean = false
) {
    val id: String = UUID.randomUUID().toString()
}

import java.util.UUID
